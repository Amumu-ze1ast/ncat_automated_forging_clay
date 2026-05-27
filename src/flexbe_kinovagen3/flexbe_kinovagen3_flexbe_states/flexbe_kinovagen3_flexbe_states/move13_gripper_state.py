#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState

from flexbe_core import EventState, Logger
from flexbe_core.proxy import ProxyActionClient, ProxySubscriberCached


class Move13RobotiqGripperCommandState(EventState):
    """
    FlexBE state to control Robotiq gripper with speed control via waypoints.
    
    -- target_position  float   Target gripper position (0.0 = open, 0.8 = closed)
    -- speed            float   Speed factor (0.1 = very slow, 1.0 = fast) - controls step delay
    -- num_steps        int     Number of intermediate steps (more = smoother, default: 10)
    -- timeout          float   Total timeout in seconds (default: 10.0)
    
    <= done                     Gripper reached target
    <= failed                   Failed to reach target or timeout
    """

    def __init__(self, target_position=0.0, speed=0.5, num_steps=10, timeout=10.0):
        super(Move13RobotiqGripperCommandState, self).__init__(
            outcomes=['done', 'failed'],
            input_keys=[],
            output_keys=[]
        )

        # Parameters
        self._target_position = float(target_position)
        self._num_steps = int(num_steps)
        self._timeout = timeout
        
        # Calculate step delay based on speed (lower speed = longer delay)
        # speed=1.0 → 0.05s per step (fast)
        # speed=0.5 → 0.1s per step (medium)
        # speed=0.1 → 0.5s per step (slow)
        self._step_delay = 0.5 / max(speed, 0.1)

        # Action client
        self._topic = '/robotiq_gripper_controller/gripper_cmd'
        self._client = ProxyActionClient({self._topic: GripperCommand})
        
        # Subscriber for joint states
        self._joint_state_topic = '/joint_states'
        self._sub = ProxySubscriberCached({self._joint_state_topic: JointState})

        # State variables
        self._start_position = None
        self._current_step = 0
        self._last_command_time = None
        self._start_time = None
        self._waiting_for_result = False

    def execute(self, userdata):
        """Execute - called periodically while state is active."""
        
        # Check overall timeout
        if self._start_time is not None:
            elapsed = self._node.get_clock().now() - self._start_time
            if elapsed.nanoseconds / 1e9 > self._timeout:
                Logger.logwarn(f'Gripper motion timed out after {self._timeout}s')
                return 'failed'

        # If waiting for previous command to complete
        if self._waiting_for_result:
            if self._client.has_result(self._topic):
                result = self._client.get_result(self._topic)
                if not result:
                    Logger.logwarn('Gripper command failed')
                    return 'failed'
                self._waiting_for_result = False
            else:
                return None  # Still waiting

        # Check if all steps completed
        if self._current_step >= self._num_steps:
            Logger.loginfo(f'✓ Gripper reached target: {self._target_position:.3f}')
            return 'done'

        # Check if enough time has passed since last command
        now = self._node.get_clock().now()
        if self._last_command_time is not None:
            elapsed = (now - self._last_command_time).nanoseconds / 1e9
            if elapsed < self._step_delay:
                return None  # Still waiting for delay

        # Calculate next waypoint position
        progress = (self._current_step + 1) / self._num_steps
        next_position = self._start_position + (self._target_position - self._start_position) * progress

        # Send command for this waypoint
        if self._send_position_command(next_position):
            self._current_step += 1
            self._last_command_time = now
            Logger.loginfo(f'Step {self._current_step}/{self._num_steps}: pos={next_position:.3f}')
        else:
            Logger.logwarn('Failed to send gripper command')
            return 'failed'

        return None

    def on_enter(self, userdata):
        """Called when state becomes active."""
        
        # Validate target position
        if not (0.0 <= self._target_position <= 0.8):
            Logger.logerr(f'Invalid target position: {self._target_position} (must be 0.0-0.8)')
            return

        # Wait for action server
        if not self._client.is_available(self._topic):
            Logger.logwarn('Waiting for gripper action server...')
            if not self._client.wait_for_server(self._topic, timeout=2.0):
                Logger.logerr('Gripper action server not available!')
                return

        # Get current gripper position
        self._start_position = self._get_current_gripper_position()
        if self._start_position is None:
            Logger.logwarn('Could not read current gripper position, using 0.0')
            self._start_position = 0.0

        # Initialize state
        self._current_step = 0
        self._last_command_time = None
        self._start_time = self._node.get_clock().now()
        self._waiting_for_result = False

        Logger.loginfo(f'Starting gripper motion: {self._start_position:.3f} → {self._target_position:.3f}')
        Logger.loginfo(f'Speed: {1.0/(self._step_delay+0.001):.1f}x, Steps: {self._num_steps}, Delay: {self._step_delay:.3f}s')

    def on_exit(self, userdata):
        """Called when state is exited."""
        if self._client.is_active(self._topic):
            self._client.cancel(self._topic)

        self._current_step = 0
        self._waiting_for_result = False

    def _get_current_gripper_position(self):
        """Get current gripper position from joint states."""
        try:
            if self._sub.has_msg(self._joint_state_topic):
                msg = self._sub.get_last_msg(self._joint_state_topic)
                
                # Look for robotiq gripper joint
                # Common names: 'robotiq_85_left_knuckle_joint', 'finger_joint', etc.
                for i, name in enumerate(msg.name):
                    if 'gripper' in name.lower() or 'finger' in name.lower() or 'knuckle' in name.lower():
                        position = msg.position[i]
                        Logger.loginfo(f'Current gripper joint "{name}": {position:.3f}')
                        return position
                
                Logger.logwarn('Could not find gripper joint in /joint_states')
        except Exception as e:
            Logger.logwarn(f'Error reading joint states: {str(e)}')
        
        return None

    def _send_position_command(self, position):
        """Send a single position command to the gripper."""
        try:
            goal = GripperCommand.Goal()
            goal.command.position = float(position)
            goal.command.max_effort = 100.0

            self._client.send_goal(self._topic, goal)
            self._waiting_for_result = True
            return True
        except Exception as e:
            Logger.logerr(f'Failed to send goal: {str(e)}')
            return False