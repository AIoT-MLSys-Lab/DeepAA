import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops

class GradualWarmup_Cosine_Scheduler(LearningRateSchedule):
    def __init__(self, starting_lr, initial_lr, ending_lr, warmup_steps, total_steps, name=None):
        super(GradualWarmup_Cosine_Scheduler, self).__init__()

        self.starting_lr = starting_lr
        self.initial_lr = initial_lr
        self.ending_lr = ending_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.name = name

    def __call__(self, step):
        with ops.name_scope_v2(self.name or 'GradualWarmup_Cosine') as name:
            initial_lr = ops.convert_to_tensor_v2(self.initial_lr, name='initial_learning_rate')
            dtype = initial_lr.dtype
            starting_lr = math_ops.cast(self.starting_lr, dtype)
            ending_lr = math_ops.cast(self.ending_lr, dtype)
            warmup_steps = math_ops.cast(self.warmup_steps, dtype)
            total_steps = math_ops.cast(self.total_steps, dtype)
            one = math_ops.cast(1.0, dtype)
            point5 = math_ops.cast(0.5, dtype)
            pi = math_ops.cast(3.1415926536, dtype)
            step = math_ops.cast(step, dtype)

            lr = tf.cond(step < warmup_steps,
                         true_fn=lambda: self._warmup_schedule(starting_lr, initial_lr, step, warmup_steps),
                         false_fn=lambda: self._cosine_annealing_schedule(initial_lr, ending_lr, step, warmup_steps, total_steps, pi,
                                                          point5, one))
        return lr

    def _warmup_schedule(self, starting_lr, initial_lr, step, warmup_steps):
        ratio = math_ops.divide(step, warmup_steps)
        lr = math_ops.add(starting_lr,
                          math_ops.multiply(initial_lr - starting_lr, ratio))
        return lr

    def _cosine_annealing_schedule(self, initial_lr, ending_lr, step, warmup_steps, total_steps, pi, point5, one):
        ratio = math_ops.divide(step - warmup_steps, total_steps - warmup_steps)
        cosine_ratio_pi = math_ops.cos(math_ops.multiply(ratio, pi))
        second_part = math_ops.multiply(point5,
                                        math_ops.multiply(initial_lr - ending_lr,
                                                          one + cosine_ratio_pi))
        lr = math_ops.add(ending_lr, second_part)
        return lr


    def get_config(self):
        return {
            'starting_lr': self.starting_lr,
            'initial_lr': self.initial_lr,
            'ending_lr': self.ending_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'name': self.name
        }