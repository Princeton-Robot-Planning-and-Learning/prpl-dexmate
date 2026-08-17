"""Park both arms at the model home, the shipping fold, or storage.

Usage (with the skill server running on the robot):

    python scripts/park_arms.py --to home --host <robot>       # session start
    python scripts/park_arms.py --to fold --host <robot>       # power-off, no grippers
    python scripts/park_arms.py --to storage --grippers ...    # power-off, grippers on

Pass ``--grippers`` once grippers are physically mounted: collision
checks then include the gripper geometry, and the shipping fold is
refused (it self-collides with grippers). Observes the arms' actual
positions, plans single-arm min-jerk moves routed through home with
every straight-line segment collision-checked in sim first, and asks
for confirmation before each motion. See ``prpl_dexmate.park`` for the
operational rules this encodes.
"""

import argparse

import numpy as np

from prpl_dexmate.motion import min_jerk_trajectory
from prpl_dexmate.park import ParkingBlocked, ParkingPlanner, clip_conf_to_limits
from prpl_dexmate.remote.client import SkillClient
from prpl_dexmate.remote.protocol import DirectiveStatus, TrajectoryDirective
from prpl_dexmate.remote.server import DEFAULT_PORT

MOVE_HZ = 20.0


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", choices=("home", "fold", "storage"), required=True)
    parser.add_argument("--host", default="192.168.0.169")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--grippers",
        action="store_true",
        help="Grippers are physically mounted: include their geometry in "
        "collision checks and refuse the shipping fold.",
    )
    args = parser.parse_args()

    client = SkillClient(args.host, args.port)
    observation = client.get_observation()
    current_right = np.array(observation.right_arm_conf)
    current_left = np.array(observation.left_arm_conf)
    print("right arm at:", np.round(current_right, 3).tolist())
    print("left arm at: ", np.round(current_left, 3).tolist())

    # Joints back-driven past their soft limits (e.g. by gripper mounting)
    # cannot legally appear in a trajectory, so plan from the clipped
    # confs and let each arm's first move pull the joint back inside via
    # a widened start-error allowance.
    current_right, overshoot_right = clip_conf_to_limits(current_right, "right_arm")
    current_left, overshoot_left = clip_conf_to_limits(current_left, "left_arm")
    start_allowance = {"right_arm": overshoot_right, "left_arm": overshoot_left}
    for component, overshoot in start_allowance.items():
        if overshoot > 0.0:
            print(
                f"WARNING: {component} is {overshoot:.3f} rad beyond a soft "
                "limit; its first move will pull the joint back into range."
            )

    print("Planning and collision-checking parking moves...")
    planner = ParkingPlanner(grippers_mounted=args.grippers)
    try:
        moves = planner.plan_parking_moves(current_right, current_left, args.to)
    except (ParkingBlocked, ValueError) as e:
        print(f"REFUSING TO MOVE: {e}")
        client.close()
        planner.close()
        raise SystemExit(1) from e
    if not moves:
        print(f"Both arms are already at {args.to}; nothing to do.")
    for move in moves:
        answer = input(
            f"\nNext: {move.component} -> {np.round(move.end, 3).tolist()} "
            f"over {move.seconds:.0f}s (max delta "
            f"{np.max(np.abs(move.end - move.start)):.2f} rad). "
            "Hand on e-stop. Execute? [y/N]: "
        )
        if answer.strip().lower() not in ("y", "yes"):
            print("Stopped by operator; arms left as they are.")
            break
        trajectory = min_jerk_trajectory(
            move.start, move.end, duration=move.seconds, hz=MOVE_HZ
        )
        max_start_error = 0.05 + start_allowance.pop(move.component, 0.0)
        result = client.execute_directive(
            TrajectoryDirective.from_array(
                move.component, trajectory, hz=MOVE_HZ, max_start_error=max_start_error
            )
        )
        print(result)
        if result.status is not DirectiveStatus.SUCCEEDED:
            print(
                f"\n*** {move.component} move {result.status.value.upper()}; "
                "stopping here — the arm is holding mid-route, NOT parked."
            )
            break
    observation = client.get_observation()
    final_right = np.array(observation.right_arm_conf)
    final_left = np.array(observation.left_arm_conf)
    print("\nfinal right:", np.round(final_right, 3).tolist())
    print("final left: ", np.round(final_left, 3).tolist())
    deviation = planner.parking_deviation(final_right, final_left, args.to)
    if deviation < 0.05:
        print(f"PARKED at {args.to} (max joint deviation {deviation:.3f} rad).")
        if args.to in ("fold", "storage"):
            print("Arms rest on end-stops: safe to power off.")
    else:
        print(
            f"*** NOT PARKED: max joint deviation from {args.to} is "
            f"{deviation:.3f} rad. Do not power off away from end-stops; "
            "investigate before proceeding."
        )
    client.close()
    planner.close()
    if deviation >= 0.05:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
