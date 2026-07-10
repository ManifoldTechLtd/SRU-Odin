"""SRU Navigation learning model (ONNX Runtime).

Identical inference pipeline to the original ROS2 deployment. No ROS deps.
"""

import cv2
import numpy as np
import onnxruntime as ort

from sru_nav_go2 import constants
from sru_nav_go2.utils import subtract_frame_transforms, transform_points


# Model architecture constants (must match training-time export).
STATE_DIM = 16              # linear_vel(3)+angular_vel(3)+gravity(3)+last_action(3)+target_pos_log(4)
DEPTH_EMBEDDING_DIM = 2560  # 64 * 5 * 8 (flattened VAE output)
POLICY_INPUT_DIM = STATE_DIM + DEPTH_EMBEDDING_DIM  # 2576
LSTM_HIDDEN_DIM = 512


class LearningModel:
    """RL navigation policy + depth VAE encoder (ONNX Runtime).

    VAE encoder:
        depth_image [1, 1, 40, 64] -> mu [1, 64, 5, 8] -> flat [2560]
    Policy (LSTM):
        obs [1, 2576] + h_in [1, 1, 512] + c_in [1, 1, 512]
        -> raw_action [1, 3], h_out, c_out
        cmd_vel = tanh(raw_action) * POLICY_SCALE
    """

    def __init__(self, preprocess_model_path, policy_model_path,
                 policy_scale=None, intra_op_num_threads=4):
        available_providers = ort.get_available_providers()

        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print('\033[92m' + 'Using device: CUDA (ONNX Runtime)' + '\033[0m')
        elif 'TensorrtExecutionProvider' in available_providers:
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                         'CPUExecutionProvider']
            print('\033[92m' + 'Using device: TensorRT (ONNX Runtime)' + '\033[0m')
        else:
            providers = ['CPUExecutionProvider']
            print('\033[93m' + 'Using device: CPU (ONNX Runtime)' + '\033[0m')

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = int(intra_op_num_threads)

        self.preprocess_session = ort.InferenceSession(
            preprocess_model_path, sess_options=sess_options, providers=providers
        )
        self.policy_session = ort.InferenceSession(
            policy_model_path, sess_options=sess_options, providers=providers
        )

        self.preprocess_input_name = self.preprocess_session.get_inputs()[0].name
        self.preprocess_output_name = self.preprocess_session.get_outputs()[0].name

        self.policy_input_names = {inp.name: inp for inp in self.policy_session.get_inputs()}
        self.policy_output_names = [out.name for out in self.policy_session.get_outputs()]

        self._h_state = np.zeros((1, 1, LSTM_HIDDEN_DIM), dtype=np.float32)
        self._c_state = np.zeros((1, 1, LSTM_HIDDEN_DIM), dtype=np.float32)

        if policy_scale is None:
            policy_scale = constants.POLICY_SCALE
        self._policy_scale = np.array(policy_scale, dtype=np.float32)

        self.fake_run_once()
        print('\033[92m' + 'Learning model is ready (ONNX Runtime).' + '\033[0m')

    # -----------------------------------------------------------------
    # Depth preprocessing
    # -----------------------------------------------------------------
    @staticmethod
    def _resize_bilinear(img, target_size):
        return cv2.resize(img, (target_size[1], target_size[0]),
                          interpolation=cv2.INTER_LINEAR)

    def depth_preprocess(self, img):
        """Encode a depth image (H, W) into a 2560-dim latent."""
        img_resized = self._resize_bilinear(img, (40, 64))
        img_tensor = img_resized.astype(np.float32)[np.newaxis, np.newaxis, :, :]
        vae_output = self.preprocess_session.run(
            [self.preprocess_output_name],
            {self.preprocess_input_name: img_tensor},
        )[0]
        return vae_output.flatten()

    # -----------------------------------------------------------------
    # Target encoding
    # -----------------------------------------------------------------
    def normalize_target_position(self, target_pos_w, robot_pos_w, robot_orientation_w):
        target_pos_w = np.array(target_pos_w, dtype=np.float32)[np.newaxis]
        robot_pos_w = np.array(robot_pos_w, dtype=np.float32)[np.newaxis]
        robot_orientation_w = np.array(robot_orientation_w, dtype=np.float32)[np.newaxis]

        inv_pos, inv_rot = subtract_frame_transforms(robot_pos_w, robot_orientation_w)
        target_vec_b = transform_points(target_pos_w, inv_pos, inv_rot)

        dist = np.linalg.norm(target_vec_b, axis=-1, keepdims=True) + 1e-6
        target_pos = target_vec_b / dist
        dist_log = np.log(dist + 1.0)

        target_pos = np.concatenate((target_pos, dist_log), axis=-1).reshape(-1)
        return target_pos, target_vec_b

    # -----------------------------------------------------------------
    # LSTM state management
    # -----------------------------------------------------------------
    def reset_hidden_state(self):
        self._h_state = np.zeros((1, 1, LSTM_HIDDEN_DIM), dtype=np.float32)
        self._c_state = np.zeros((1, 1, LSTM_HIDDEN_DIM), dtype=np.float32)

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------
    def predict(self, linear_vel, angular_vel, gravity_vector, last_action,
                target_pos_w, robot_pos_w, robot_orientation_w, depth_image,
                is_reset=False):
        if is_reset:
            self.reset_hidden_state()

        depth_embedding = self.depth_preprocess(depth_image)

        target_pos_log, target_vec_b = self.normalize_target_position(
            target_pos_w, robot_pos_w, robot_orientation_w
        )

        state_input = np.array(
            list(linear_vel) + list(angular_vel) + list(gravity_vector) +
            list(last_action) + target_pos_log.tolist(),
            dtype=np.float32,
        )

        obs = np.concatenate([state_input, depth_embedding])[np.newaxis].astype(np.float32)

        outputs = self.policy_session.run(
            None,
            {
                'obs': obs,
                'h_in': self._h_state,
                'c_in': self._c_state,
            },
        )

        raw_action, h_out, c_out = outputs
        self._h_state = h_out
        self._c_state = c_out

        cmd_vel = np.tanh(raw_action) * self._policy_scale
        cmd_vel = cmd_vel.squeeze(0)
        raw_action = raw_action.squeeze(0)

        return cmd_vel, raw_action, target_vec_b

    def fake_run_once(self):
        linear_vel = [0.0, 0.0, 0.0]
        angular_vel = [0.0, 0.0, 0.0]
        gravity_vector = [0.0, 0.0, -1.0]
        target_pos_w = [1.0, 0.0, 0.0]
        robot_pos_w = [0.0, 0.0, 0.0]
        robot_orientation_w = [1.0, 0.0, 0.0, 0.0]
        last_action = [0.0, 0.0, 0.0]
        depth_image = np.random.rand(600, 960).astype(np.float32)

        cmd_vel, _, _ = self.predict(
            linear_vel, angular_vel, gravity_vector, last_action,
            target_pos_w, robot_pos_w, robot_orientation_w, depth_image,
            is_reset=True,
        )
        print('Warmup predicted cmd_vel: linear_x={:.4f}, linear_y={:.4f}, '
              'angular_z={:.4f}'.format(cmd_vel[0], cmd_vel[1], cmd_vel[2]))
        self.reset_hidden_state()
