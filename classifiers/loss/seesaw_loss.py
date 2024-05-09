"""Custom losses"""

import tensorflow as tf

# pylint: disable = attribute-defined-outside-init, no-name-in-module, unexpected-keyword-arg
# pylint: disable = no-value-for-parameter
from tensorflow.python.ops import state_ops as tf_state_ops


class SeeSawWeightCalculator(tf.keras.layers.Layer):
    """custom layer for calculating seesaw loss weight factors"""

    def __init__(
        self, num_classes: int = 10, p_factor: float = 0.8, q_factor: float = 2.0
    ) -> None:
        """[summary]
        Args:
            num_classes (int, optional): Total # of categories. Defaults to 10.
            p_factor (float, optional): mitigation factors tuning parameter. Defaults to 0.8.
            q_factor (float, optional): compensation factors tuning parameter. Defaults to 2.0.
        """
        super().__init__(trainable=False)
        self.num_classes = num_classes
        self.p_factor = p_factor
        self.q_factor = q_factor

    def build(self, input_shape):
        """build the layer"""
        shape = (1, self.num_classes)
        self.freq_accumulator = self.add_weight(
            shape=shape,
            name="class_freqs",
            initializer=tf.ones_initializer(),
            trainable=False,
        )
        return super().build(input_shape)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """call method on the layer
        Args:
            inputs (tf.Tensor): sample wise loss values for a given batch
        Returns:
            tf.Tensor (shape = ()): loss threshold value for importance sampling
        """
        seesaw_weights = tf.ones_like(y_true)

        if self.p_factor > 0:
            seesaw_weights *= self._get_mitigator_factor(y_true)
        if self.q_factor > 0:
            seesaw_weights *= self._get_compensation_factor(y_true, y_pred)

        return seesaw_weights

    def _get_mitigator_factor(self, labels: tf.Tensor) -> tf.Tensor:
        """Calcualtes the mitigation factors for seesaw loss
        Args:
            labels (tf.Tensor): ground truth labels for the classification task
        Returns:
            tf.Tensor: mitigation factors
        """

        self.freq_accumulator = tf_state_ops.assign(
            self.freq_accumulator,
            self.freq_accumulator + tf.reduce_sum(labels, axis=0, keepdims=True),
        )
        freq_comparator = tf.tile(self.freq_accumulator, [self.num_classes, 1])
        mask = tf.cast(
            tf.greater(tf.transpose(self.freq_accumulator), freq_comparator),
            dtype=labels.dtype,
        )
        long_tail = freq_comparator / tf.transpose(self.freq_accumulator)
        mitigator = (long_tail ** self.p_factor) * mask + (1 - mask)
        return tf.gather(mitigator, tf.argmax(labels, axis=-1))

    def _get_compensation_factor(
        self, labels: tf.Tensor, logits: tf.Tensor
    ) -> tf.Tensor:
        """Calcualtes the mitigation factors for seesaw loss
        Args:
            labels (tf.Tensor): one-hot encoded ground truth labels for the classification task
            logits (tf.Tensor): predicted logits for the classification task
        Returns:
            tf.Tensor: mitigation factors
        """
        scores = tf.math.softmax(logits, axis=-1)
        gt_label = tf.argmax(labels, axis=-1)
        indices = tf.stack(
            [tf.range(0, tf.shape(logits)[0], dtype=gt_label.dtype), gt_label], axis=1
        )
        cls_score = tf.gather_nd(scores, indices)
        compensator = scores / tf.expand_dims(cls_score, axis=-1)
        mask = tf.cast(tf.greater(compensator, 1.0), dtype=labels.dtype)
        return (compensator ** self.q_factor) * mask + (1 - mask)

    def get_config(self) -> dict:
        """Setting up the layer config
        Returns:
            dict: config key-value pairs
        """
        base_config = super().get_config()
        config = {
            "num_classes": self.num_classes,
            "freq_accumulator": self.freq_accumulator,
        }
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """shape of the layer output"""
        return input_shape


class SeeSawLoss(tf.keras.losses.Loss):
    """
    Custom seesaw loss based on
    "Seesaw Loss for Long-Tailed Instance Segmentation (https://arxiv.org/abs/2008.10032)"
    """

    def __init__(
        self,
        num_classes: int = 2,
        p_factor: float = 0.8,
        q_factor: float = 2.0,
        axis: int = -1,
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        **kwargs,
    ) -> None:
        """
        Args:
            num_classes (int, optional): # of classes in the whole dataset. Defaults to 10.
            p_factor (float, optional): mitigation factors tuning parameter. Defaults to 0.8.
            q_factor (float, optional): compensation factors tuning parameter. Defaults to 2.0.
            axis (int, optional): Defaults to -1.
            reduction ([type], optional): Defaults to tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes

        self.axis = axis
        self.reduction = reduction
        self.weight_calculator = SeeSawWeightCalculator(
            num_classes, p_factor=p_factor, q_factor=q_factor
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Args:
            y_true (tf.Tensor): one-hot encoded ground truth labels
            y_pred (tf.Tensor): predicted logits
        Returns:
            tf.Tensor:
        """

        # get the mitigation factors
        weights = self.weight_calculator(y_true, y_pred)
        logits = y_pred + tf.math.log(weights) * (1 - y_true)

        return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)