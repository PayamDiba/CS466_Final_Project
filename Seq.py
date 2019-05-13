import numpy as np
import pandas as pd


class Seq(object):
    def __init__(self, seq_file, expression_file, TF_file, nEnhancers, nBins, nTrain, nValid, nTest):
        """
        seq_file: A FASTA formatted sequence file
        expression_file: expression file of each enhancer in 17 bins
        TF_file: expression of TFs in 17 bins

        nTrain: first nTrain enhancers are used for training
        nValid: next nValid enhancers are used for validation
        nTest: last nTest enhancers used for testing

        nEnhancers: nTrain, nValid and nTest should sum to nEnhancers
        nBins: number of points on V-D axis for which the expression of enhancer was measured
        """

        self.nEnhancers_ = nEnhancers
        self.nBins_ = nBins
        # This stores the starting data ID for next batch
        self.nextBatch_startID = 0


        # has three main keywords: train, valid, and test --> each one is mapped to a second dictionary
        #second disctionary maps data_point_id to another dictionary, third dictionary has following
        #keys:
        #'seq'
        #'seq_encoded'
        #'E_rho'
        #'E_TFs'
        self.data = {}
        self.data['train'] = {}
        self.data['valid'] = {}
        self.data['test'] = {}
        self.nTrain_data = nTrain * self.nBins_
        self.shuffleIDS()
        #####################################################################



        seqLabels, allSeq = self.read_seq_file(seq_file)
        expr_seqLabels, seqExpr = self.read_expr_file(expression_file)
        tfExpr = self.read_TF_file(TF_file)

        ID_tarin = 0
        ID_valid = 0
        ID_test = 0
        # The first 30 enhancers are used for training, the last 8 are for cross validation
        for si in range(self.nEnhancers_):
            for bi in range(self.nBins_):
                currSeq = allSeq[si]

                ## Check consistency in the input files
                if seqLabels[si] != expr_seqLabels[si]:
                    #ToDo: handle this case in the code
                    #print seqLabels[si]
                    #print expr_seqLabels[si]
                    raise ValueError('Input files are inconsistent, use the same order for sequences in the input sequence and expression files')

                if si < nTrain:
                    self.data['train'][ID_tarin] = {}
                    self.data['train'][ID_tarin]['seq'] = currSeq
                    self.data['train'][ID_tarin]['seq_encoded'] = self.encode_seq(currSeq)
                    self.data['train'][ID_tarin]['E_rho'] = seqExpr[si, bi]
                    self.data['train'][ID_tarin]['E_TFs'] = tfExpr[:,bi]
                    ID_tarin += 1
                elif si < nTrain + nValid:
                    self.data['valid'][ID_valid] = {}
                    self.data['valid'][ID_valid]['seq'] = currSeq
                    self.data['valid'][ID_valid]['seq_encoded'] = self.encode_seq(currSeq)
                    self.data['valid'][ID_valid]['E_rho'] = seqExpr[si, bi]
                    self.data['valid'][ID_valid]['E_TFs'] = tfExpr[:,bi]
                    ID_valid += 1
                else:
                    self.data['test'][ID_test] = {}
                    self.data['test'][ID_test]['seq'] = currSeq
                    self.data['test'][ID_test]['seq_encoded'] = self.encode_seq(currSeq)
                    self.data['test'][ID_test]['E_rho'] = seqExpr[si, bi]
                    self.data['test'][ID_test]['E_TFs'] = tfExpr[:,bi]
                    ID_test += 1

    def shuffleIDS(self):
        ids = np.arange(self.nTrain_data)
        np.random.shuffle(ids)
        self.shuffled_id_ = ids

    def read_seq_file (self, seq_file, length = 332):
        """
        This method reads all the sequences from the sequence file and add 'N' to sequences
        that have a shorter length than the specified length. 'N' will be encoded to
        an all-zero column
        """
        df = pd.read_csv(seq_file, header=None, index_col=None, sep='\n')
        df = df.values
        allSeq = []
        seqLabel = []

        for l in df:
            currLine = l[0]

            if currLine[0] != '>':
                if len(currLine) < length:
                    d = length - len(currLine)
                    for i in range(d):
                        currLine = 'N' + currLine
                    allSeq.append(currLine)

                else:
                    allSeq.append(currLine)
            else:
                seqLabel.append(currLine[1:])

        return seqLabel, allSeq


    def read_expr_file(self, expr_file):
        df = pd.read_csv(expr_file, header=0, index_col=0, sep='\t')
        label = df.axes[0].values
        expr = df.values

        return label, expr

    def read_TF_file(self, TF_file):
        df = pd.read_csv(TF_file, header=0, index_col=0, sep='\t')
        expr = df.values

        return expr

    def encode_seq(self, seq_string):
        """
        Input: one DNA sequence as string of length L
        returns: np.array(L,4), one-hot encoded sequence

        4 columns: A, C, G, T
        """

        ret = np.zeros((len(seq_string),4))
        for ib, b in enumerate(seq_string):
            if b.capitalize() == 'A':
                ret[ib, 0] = 1
            elif b.capitalize() == 'C':
                ret[ib, 1] = 1
            elif b.capitalize() == 'G':
                ret[ib, 2] = 1
            elif b.capitalize() == 'T':
                ret[ib, 3] = 1

            # else b == 'N': let it remain zero

        return ret

    def next_batch(self, all_data = None, size = 20):
        """
        all_data : None, 'train', 'test', 'all'
        if all_data == None:
            This method returns a batch of training data of the given size if all_data == None.

        if all_data == 'train':
            returns all of the training data

        if all_data == 'test'
            returns all of the test data

        if all_data == 'valid'
            returns all of the validation data

        if all_data == 'all'
            return all of the training and test data

        Returns:
            sequence = np.arrays(batch_size * L * 4)
            TF_concentration = np.array(batch_size,3)
            rho_expression = np.array(batch_size,)
        """
        sequence = []
        TF_concentration = []
        rho_expression = []

        if all_data == None:
            lenTrainData = len(self.data['train'].keys())

            idx = self.nextBatch_startID
            for i in range(size):
                if idx < lenTrainData:
                    id_shuffled = self.shuffled_id_[idx]
                    currSeq = self.data['train'][id_shuffled]['seq_encoded'].tolist()
                    currTF_conc = self.data['train'][id_shuffled]['E_TFs'].tolist()
                    currRho_expr = self.data['train'][id_shuffled]['E_rho'].tolist()

                    sequence.append(currSeq)
                    TF_concentration.append(currTF_conc)
                    rho_expression.append(currRho_expr)

                    idx += 1
                else:
                    break

            if idx == lenTrainData:
                self.nextBatch_startID = 0
                self.shuffleIDS()
            elif idx < lenTrainData:
                self.nextBatch_startID = idx
            else:
                raise ValueError('Something is wrong')

            return np.array(sequence), np.array(TF_concentration), np.array(rho_expression)

        elif all_data == 'train':
            lenTrainData = len(self.data['train'].keys())

            for ID in range(lenTrainData):
                currSeq = self.data['train'][ID]['seq_encoded'].tolist()
                currTF_conc = self.data['train'][ID]['E_TFs'].tolist()
                currRho_expr = self.data['train'][ID]['E_rho'].tolist()

                sequence.append(currSeq)
                TF_concentration.append(currTF_conc)
                rho_expression.append(currRho_expr)

            return np.array(sequence), np.array(TF_concentration), np.array(rho_expression)

        elif all_data == 'test':
            lenTestData = len(self.data['test'].keys())

            for ID in range(lenTestData):
                currSeq = self.data['test'][ID]['seq_encoded'].tolist()
                currTF_conc = self.data['test'][ID]['E_TFs'].tolist()
                currRho_expr = self.data['test'][ID]['E_rho'].tolist()

                sequence.append(currSeq)
                TF_concentration.append(currTF_conc)
                rho_expression.append(currRho_expr)

            return np.array(sequence), np.array(TF_concentration), np.array(rho_expression)

        elif all_data == 'valid':
            lenValidData = len(self.data['valid'].keys())

            for ID in range(lenValidData):
                currSeq = self.data['valid'][ID]['seq_encoded'].tolist()
                currTF_conc = self.data['valid'][ID]['E_TFs'].tolist()
                currRho_expr = self.data['valid'][ID]['E_rho'].tolist()

                sequence.append(currSeq)
                TF_concentration.append(currTF_conc)
                rho_expression.append(currRho_expr)

            return np.array(sequence), np.array(TF_concentration), np.array(rho_expression)

        elif all_data == 'all':
            lenTrainData = len(self.data['train'].keys())
            lenTestData = len(self.data['test'].keys())
            lenValidData = len(self.data['valid'].keys())

            for ID in range(lenTrainData):
                currSeq = self.data['train'][ID]['seq_encoded'].tolist()
                currTF_conc = self.data['train'][ID]['E_TFs'].tolist()
                currRho_expr = self.data['train'][ID]['E_rho'].tolist()

                sequence.append(currSeq)
                TF_concentration.append(currTF_conc)
                rho_expression.append(currRho_expr)

            for ID in range(lenValidData):
                currSeq = self.data['valid'][ID]['seq_encoded'].tolist()
                currTF_conc = self.data['valid'][ID]['E_TFs'].tolist()
                currRho_expr = self.data['valid'][ID]['E_rho'].tolist()

                sequence.append(currSeq)
                TF_concentration.append(currTF_conc)
                rho_expression.append(currRho_expr)

            for ID in range(lenTestData):
                currSeq = self.data['test'][ID]['seq_encoded'].tolist()
                currTF_conc = self.data['test'][ID]['E_TFs'].tolist()
                currRho_expr = self.data['test'][ID]['E_rho'].tolist()

                sequence.append(currSeq)
                TF_concentration.append(currTF_conc)
                rho_expression.append(currRho_expr)

            return np.array(sequence), np.array(TF_concentration), np.array(rho_expression)
