import numpy as np
import csv
import pandas as pd
import os
from Expression_Model import Expression_Model
from Seq import Seq
from absl import flags
from absl import logging
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 0.001, 'learning rate of gradient descent')
flags.DEFINE_integer('bs', 35, 'batch size used for training')
flags.DEFINE_integer('ns', None, 'total number of training steps (total number of batches used for training)')
flags.DEFINE_integer('nib', 1, 'number of training step on each batch')
flags.DEFINE_float('ds', None, 'Drop rate of sequence')
flags.DEFINE_float('dm', None, 'Drop rate of motifs')
flags.DEFINE_float('rm', None, 'L2 regularization scale of motifs')
flags.DEFINE_float('rc', None, 'L2 regularization scale of cooperativity filters')
flags.DEFINE_float('rnn', None, 'L2 regularization scale of NN')
flags.DEFINE_string('otf', None, 'output file name (path) for tensorflow params and checkpoint')
flags.DEFINE_string('otr', None, 'output file name (path) for training performance')
flags.DEFINE_string('ote', None, 'output file name (path) for test performance')
flags.DEFINE_string('ov', None, 'output file name (path) for validation performance')
flags.DEFINE_string('op', None, 'output file name (path) for parameters dictionary')

def main(argv):
    """
    First makes sure that all directories exists. This might cause problem, make sure it works fine
    """
    ## the following lines for making directory only works in python 3
    for filename in [FLAGS.otf, FLAGS.otr, FLAGS.ote, FLAGS.ov, FLAGS.op]:
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        except:
            pass

    S = Seq('seqs2.fa','expr2.tab','factor_expr.tab',49,17, 32, 6, 11)
    model = Expression_Model(332)
    model.train(S, learning_rate = FLAGS.lr, batch_size = FLAGS.bs, num_steps = FLAGS.ns, drop_rate_seq = FLAGS.ds, drop_rate_motifs = FLAGS.dm, reg_scale_motifs = FLAGS.rm, reg_scale_coops = FLAGS.rc, reg_scale_NN = FLAGS.rnn, num_iter_batch = FLAGS.nib, print_performance = False, output_name_Tensorflow = FLAGS.otf, output_name_train_err = FLAGS.otr, output_name_valid_err = FLAGS.ov, output_name_test_err = FLAGS.ote, output_name_pars_dict = FLAGS.op)

if __name__ == "__main__":
    app.run(main)
