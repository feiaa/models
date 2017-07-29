#! coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import io
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def word_embedding(LOG_DIR, emb, tsv):
    """Embedding visualization using tensorboard.
    
    Args:
        LOG_DIR: 
        emb: The embedding tensor with shape n x d which each line representing a vector of a word.
        tsv: Table separated values. If the meta in tsv only one column, then the number of lines should equals n. Else 
            tsv should have n + 1 lines with the first line contains the header.
            If tsv is a string, then it represents the file path where the tsv file resides. Other wise it should be a list.
    """
    # Create randomly initialized embedding weights which will be trained.
    g = tf.Graph()
    with g.as_default():

        emb = np.array(emb)
        n, d = emb.shape
        embedding_var = tf.Variable(emb, dtype=tf.float32, name='word_embedding')

        # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
        config = projector.ProjectorConfig()

        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        # Link this tensor to its metadata file (e.g. labels).
        if isinstance(tsv, str):
            embedding.metadata_path = os.path.join(LOG_DIR, tsv)
        else:
            meta_path = os.path.join(LOG_DIR, "_meatadata.tsv")
            with io.open(meta_path, "w") as f:
                for line in tsv:
                    f.write(line + '\n')
            embedding.metadata_path = meta_path
        # Use the same LOG_DIR where you stored your checkpoint.
        summary_writer = tf.summary.FileWriter(LOG_DIR)

        # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
        # read this file during startup.
        projector.visualize_embeddings(summary_writer, config)

        with tf.Session(graph=g) as sess:
            tf.global_variables_initializer().run()
            tf.train.Saver().save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=1)


if __name__ == "__main__":

    with tf.Session() as sess:
        n, d = 1000, 200
        emb = tf.random_normal([n, d])
        tsv = [str(i) for i in range(n)]
        word_embedding('/tmp/log', emb.eval(), tsv)

# tesorboard --logdir=/tmo/log

