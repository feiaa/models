#! coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class ScalarLogger(object):

    def __init__(self, log_dir, summary_names):
        self.log_path = log_dir
        # use _name for tensor placeholder, and name for summary
        for name in summary_names:
            setattr(self, '_' + name, tf.placeholder(dtype=tf.float32, shape=()))
            setattr(self, name, tf.summary.scalar(name, getattr(self, '_' + name)))
        self.writer = tf.summary.FileWriter(log_dir)
        self.allowed_names = summary_names

    def __write_single_summary(self, sess, name, value, global_step):
        if not (name in self.allowed_names):
            raise ValueError("%s not in allowed summaries" % name)
        self.writer.add_summary(sess.run(getattr(self, name), feed_dict={getattr(self, '_' + name) : value}), global_step=global_step)

    def summary(self, sess, names, values, global_step):
        """Write summary using FileWriter.
        
        Args:
            names: If names is a list, then write all names in the list. Else if names is a string, then it is a single 
                summary name. the name should in self.allow names.
            values: The value corresponding to names.
        """
        if isinstance(names, str):
            self.__write_single_summary(sess, names, values, global_step)
        elif isinstance(names, list):
            for name, value in zip(names, values):
                self.__write_single_summary(sess, name, value, global_step)
        else:
            raise ValueError("Wrong typed names.")

# Usage examples.
if __name__ == "__main__":
    with tf.Session() as sess:
        logger = ScalarLogger("/tmp/my_log", ['v1', 'v2', 'v3'])
        for i in range(100):
            logger.summary(sess, ['v1', 'v2'], [i, 2 * i], i)
        for i in range(100):
            logger.summary(sess, 'v3', i, i)

# tesorboard --logdir=/tmo/my_log
