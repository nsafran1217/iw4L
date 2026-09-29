//! Stepping the listen authority on a worker thread (`IW4L_AUTHORITY_THREAD=1`).
//!
//! The fixed loop still runs Advance, Ingress and Gather for every tick on the
//! main thread, but instead of stepping it records the tick
//! ([`defer_authority_step`]). After `ClientSet::Load` the authority's
//! `SimWorld` and the recorded ticks go to a worker thread, which steps them
//! while the main thread runs `Receive` through `Effects`. Before
//! `ClientSet::Diag` the main thread takes the world back and publishes each
//! stepped tick by running `FixedUpdate` again in publish mode, where only the
//! Snapshot, Fanout and Bookkeeping sets run. Design and trade-offs:
//! `handoff/AUTHORITY-THREAD-DESIGN.md` in the ia64 work tree.

use std::sync::{Mutex, mpsc};

use bevy::prelude::*;
use frame::{ExitLevelCalled, MatchTornDown};
use sim::ClientId;

use crate::authority::inbox::AuthorityClock;
use crate::authority::runtime::{
    AuthorityWorld, ClientShotSamples, PendingAcks, PendingAuthorityInput, PendingStepResult,
    StepOutcome, report_step_fault, step_world,
};
use crate::client::predict::CmdSeq;
use crate::schedule::{AuthoritySet, ClientSet};

const AUTHORITY_THREAD_ENV: &str = "IW4L_AUTHORITY_THREAD";

/// Whether the authority steps on the worker thread. Set once at startup.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuthorityThreadMode {
    pub enabled: bool,
}

impl AuthorityThreadMode {
    /// On when `IW4L_AUTHORITY_THREAD` is set, for a listen game without the
    /// master bridge (the only arrangement it has been designed and tested for).
    pub fn from_env(role: frame::RuntimeRole, master_enabled: bool) -> Self {
        let requested =
            std::env::var(AUTHORITY_THREAD_ENV).is_ok_and(|v| !v.is_empty() && v != "0");
        let enabled = requested && role == frame::RuntimeRole::Listen && !master_enabled;
        if requested && !enabled {
            diag::warn!(
                Net,
                "{AUTHORITY_THREAD_ENV}: ignored (only a listen game without the master bridge)"
            );
        }
        Self { enabled }
    }
}

/// True while `FixedUpdate` runs to publish a tick the worker stepped.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuthorityPublishing(pub bool);

/// The authority clock of the newest tick published to the client. Kept only
/// with the worker thread, where it trails the live clock by the ticks still
/// being stepped; inline the live clock is the published one.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct PublishedAuthorityClock(pub Option<AuthorityClock>);

/// The system set holding the launch, so systems that touch `AuthorityWorld`
/// outside the client sets can order themselves before it.
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AuthorityLaunchSet;

/// What the main thread keeps of a deferred tick to publish it.
struct TickMeta {
    clock: AuthorityClock,
    acks: Vec<(ClientId, CmdSeq)>,
}

/// What the worker needs to step a deferred tick.
struct TickWork {
    tick: u32,
    input: sim::TickInput,
    samples: Vec<((ClientId, i32), sim::ShotSampleProvenance)>,
}

#[derive(Resource, Default)]
struct DeferredBatch {
    meta: Vec<TickMeta>,
    work: Vec<TickWork>,
}

struct Job {
    world: sim::SimWorld,
    work: Vec<TickWork>,
}

struct Done {
    world: sim::SimWorld,
    outcomes: Vec<StepOutcome>,
}

#[derive(Resource)]
struct AuthorityWorker {
    jobs: mpsc::Sender<Job>,
    done: Mutex<mpsc::Receiver<std::thread::Result<Done>>>,
    in_flight: Option<Vec<TickMeta>>,
}

impl AuthorityWorker {
    fn spawn() -> Self {
        let (jobs, job_rx) = mpsc::channel::<Job>();
        let (done_tx, done) = mpsc::channel();
        std::thread::Builder::new()
            .name("authority".into())
            .spawn(move || {
                for job in job_rx {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run_job(job)
                    }));
                    if done_tx.send(result).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn the authority worker thread");
        Self {
            jobs,
            done: Mutex::new(done),
            in_flight: None,
        }
    }
}

fn run_job(job: Job) -> Done {
    let Job { mut world, work } = job;
    let mut outcomes = Vec::with_capacity(work.len());
    for tick in work {
        let _span = perf::Span::AuthorityWorkerStep.enter();
        let outcome = step_world(&mut world, tick.tick, tick.input, tick.samples);
        // A failed step freezes the world, as it does inline: the rest of the
        // batch is not stepped.
        let failed = matches!(outcome, StepOutcome::Fault(_));
        outcomes.push(outcome);
        if failed {
            break;
        }
    }
    Done { world, outcomes }
}

pub(crate) fn register(app: &mut App, mode: AuthorityThreadMode) {
    app.insert_resource(mode)
        .init_resource::<AuthorityPublishing>()
        .init_resource::<PublishedAuthorityClock>();
    // Inline (the default) runs every set every fixed tick; publish runs happen
    // only with the worker thread.
    app.configure_sets(
        FixedUpdate,
        (
            AuthoritySet::Advance.run_if(not_publishing),
            AuthoritySet::Ingress.run_if(not_publishing),
            AuthoritySet::Gather.run_if(not_publishing),
            AuthoritySet::Step.run_if(not_publishing),
            AuthoritySet::Snapshot.run_if(inline_or_publishing),
            AuthoritySet::Fanout.run_if(inline_or_publishing),
            AuthoritySet::Bookkeeping.run_if(inline_or_publishing),
        ),
    );
    if !mode.enabled {
        return;
    }
    app.init_resource::<DeferredBatch>()
        .insert_resource(AuthorityWorker::spawn())
        .add_systems(
            FixedUpdate,
            defer_authority_step
                .in_set(AuthoritySet::Step)
                .run_if(crate::authority_should_tick),
        )
        .configure_sets(
            Update,
            AuthorityLaunchSet
                .after(ClientSet::Load)
                .before(ClientSet::Receive),
        )
        .add_systems(
            Update,
            (
                drop_deferred_on_match_torn_down.in_set(ClientSet::Load),
                launch_authority_batch.in_set(AuthorityLaunchSet),
                join_authority_batch
                    .after(ClientSet::Effects)
                    .before(ClientSet::Diag),
            ),
        );
    diag::info!(Net, "authority: stepping on a worker thread ({AUTHORITY_THREAD_ENV})");
}

pub fn authority_thread_enabled(mode: Option<Res<AuthorityThreadMode>>) -> bool {
    mode.is_some_and(|mode| mode.enabled)
}

pub fn authority_inline(mode: Option<Res<AuthorityThreadMode>>) -> bool {
    !authority_thread_enabled(mode)
}

fn not_publishing(publishing: Option<Res<AuthorityPublishing>>) -> bool {
    !publishing.is_some_and(|p| p.0)
}

fn inline_or_publishing(
    mode: Option<Res<AuthorityThreadMode>>,
    publishing: Option<Res<AuthorityPublishing>>,
) -> bool {
    authority_inline(mode) || publishing.is_some_and(|p| p.0)
}

/// The Step set with the worker thread: keep the tick for the worker instead of
/// stepping it.
fn defer_authority_step(
    clock: Res<AuthorityClock>,
    mut pending: ResMut<PendingAuthorityInput>,
    mut acks: ResMut<PendingAcks>,
    samples: Res<ClientShotSamples>,
    mut batch: ResMut<DeferredBatch>,
) {
    let Some(input) = pending.0.take() else {
        return;
    };
    batch.meta.push(TickMeta {
        clock: *clock,
        acks: std::mem::take(&mut acks.0),
    });
    batch.work.push(TickWork {
        tick: clock.tick,
        input,
        samples: samples.0.iter().map(|(key, sample)| (*key, *sample)).collect(),
    });
}

/// Ticks deferred by a fixed loop that ran before the match was torn down
/// belong to the old world, and so does the published clock.
fn drop_deferred_on_match_torn_down(
    mut torn: MessageReader<MatchTornDown>,
    mut batch: ResMut<DeferredBatch>,
    mut published: ResMut<PublishedAuthorityClock>,
) {
    if torn.read().count() > 0 {
        *batch = DeferredBatch::default();
        published.0 = None;
    }
}

fn launch_authority_batch(world: &mut World) {
    if world.resource::<DeferredBatch>().work.is_empty() {
        return;
    }
    let batch = std::mem::take(&mut *world.resource_mut::<DeferredBatch>());
    let Some(AuthorityWorld(sim_world)) = world.remove_resource::<AuthorityWorld>() else {
        diag::warn!(Net, "authority: {} deferred ticks without a world", batch.work.len());
        return;
    };
    let mut worker = world.resource_mut::<AuthorityWorker>();
    worker
        .jobs
        .send(Job {
            world: sim_world,
            work: batch.work,
        })
        .expect("the authority worker thread is gone");
    worker.in_flight = Some(batch.meta);
}

fn join_authority_batch(world: &mut World) {
    let Some(meta) = world.resource_mut::<AuthorityWorker>().in_flight.take() else {
        return;
    };
    let result = {
        let _span = perf::Span::AuthorityJoinWait.enter();
        world
            .resource::<AuthorityWorker>()
            .done
            .lock()
            .expect("authority worker channel")
            .recv()
            .expect("the authority worker thread is gone")
    };
    let done = match result {
        Ok(done) => done,
        Err(panic) => std::panic::resume_unwind(panic),
    };
    world.insert_resource(AuthorityWorld(done.world));
    let _span = perf::Span::AuthorityPublish.enter();
    for (meta, outcome) in meta.into_iter().zip(done.outcomes) {
        publish_tick(world, meta, outcome);
    }
}

/// Runs `FixedUpdate` in publish mode for one stepped tick, with the tick's own
/// clock and acks in place as they would have been inline.
fn publish_tick(world: &mut World, meta: TickMeta, outcome: StepOutcome) {
    let tick = match outcome {
        StepOutcome::Stepped(tick) => tick,
        StepOutcome::Fault(fault) => {
            if report_step_fault(fault) {
                world.write_message(ExitLevelCalled);
            }
            return;
        }
    };
    world.resource_scope(|world, mut svc: Mut<crate::PendingSvcSounds>| {
        svc.occupy_cs(&mut world.resource_mut::<AuthorityWorld>().0);
    });
    world.resource_mut::<PendingStepResult>().0 = Some(tick);
    world.resource_mut::<PendingAcks>().0 = meta.acks;
    let live = std::mem::replace(&mut *world.resource_mut::<AuthorityClock>(), meta.clock);
    world.resource_mut::<AuthorityPublishing>().0 = true;
    world.run_schedule(FixedUpdate);
    world.resource_mut::<AuthorityPublishing>().0 = false;
    *world.resource_mut::<AuthorityClock>() = live;
    world.resource_mut::<PublishedAuthorityClock>().0 = Some(meta.clock);
}
