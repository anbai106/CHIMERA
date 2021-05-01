import sys, csv, os
import pickle
from .optimization_utils import *
from sklearn.metrics import adjusted_rand_score as ARI
import pandas as pd

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen, Aoyan Dong"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Maintaining"

def clustering_main(dataFile, covariate_tsv, output_dir, config):
    """Core function of CHIMERA, performs:
        1) read and preprocess data
        2) clustering
        3) save results
    """
    outFile = os.path.join(output_dir, 'clustering_assignment.tsv')
    #================================= Reading Data ======================================================
    print('\treading data...\n')
    ### read the input tsv
    df_feature = pd.read_csv(dataFile, sep='\t')
    if ('participant_id' != list(df_feature.columns.values)[0]) or (
            'session_id' != list(df_feature.columns.values)[1]) or \
            ('diagnosis' != list(df_feature.columns.values)[2]):
        raise Exception("the data file is not in the correct format."
                        "Columns should include ['participant_id', 'session_id', 'diagnosis']")
    subjects = list(df_feature['participant_id'])
    sessions = list(df_feature['session_id'])
    diagnosis = list(df_feature['diagnosis'])
    ID = df_feature['participant_id'].to_numpy().astype(str)
    group = df_feature['diagnosis'].to_numpy().astype(int)
    feat_img = df_feature.to_numpy()[:, 3:].astype(float)

    df_covariate = pd.read_csv(covariate_tsv, sep='\t')
    if ('participant_id' != list(df_feature.columns.values)[0]) or (
            'session_id' != list(df_feature.columns.values)[1]) or \
            ('diagnosis' != list(df_feature.columns.values)[2]):
        raise Exception("the data file is not in the correct format."
                        "Columns should include ['participant_id', 'session_id', 'diagnosis']")
    participant_covariate = list(df_covariate['participant_id'])
    session_covariate = list(df_covariate['session_id'])
    label_covariate = list(df_covariate['diagnosis'])

    # check that the feature_tsv and covariate_tsv have the same orders for the first three column
    if (not subjects == participant_covariate) or (not sessions == session_covariate) or (
            not diagnosis == label_covariate):
        raise Exception(
            "the first three columns in the feature csv and covariate csv should be exactly the same.")

    ### check if site exists in covariate_tsv file
    if 'site' not in list(df_covariate.columns.values):
        config['r'] = 0
        feat_set = None
    else:
        feat_set = df_covariate['site'].to_numpy()
    del df_covariate['site']
    feat_cov = df_covariate.iloc[:, 3:].to_numpy()

    #================================= Normalizing Data ======================================================
    model, feat_img, feat_cov = data_normalization(feat_img, feat_cov, config)

    #================================= Prepare Dataset ID ======================================================
    if feat_set is None:
        config['rs'] = 0
    else:
        unique_ID = np.unique(feat_set)
        datasetID = np.copy(feat_set)
        feat_set = np.zeros((len(datasetID), len(unique_ID)))
        for i in range(len(unique_ID)):
            feat_set[np.nonzero(datasetID == unique_ID[i])[0], i] = 1 ## one-hot

    #================================= Calculate auto weight ==================================================
    if feat_cov is None:
        config['r'] = 0
    else:
        ## update the covariate distance
        config['r'] = np.sum(np.var(feat_cov, axis=0))/np.sum(np.var(feat_img, axis=0))

    #================================= Verbose information ==================================================
    if config['verbose']:
        sys.stdout.write('\t\t================= data summary ==================\n')
        sys.stdout.write('\t\tnumber of patients: %d\n' % sum(group==1))
        sys.stdout.write('\t\tnumber of normal controls: %d\n' % sum(group==0))
        sys.stdout.write('\t\timaging feature dimension: %d\n' % feat_img.shape[1])
        if feat_cov is not None:
            sys.stdout.write('\t\tcovariates dimension: %d\n' % feat_cov.shape[1])
        if feat_set is not None:
            sys.stdout.write('\t\tunique data set id: %d\n' % len(unique_ID))
        sys.stdout.write('\t\t================ configurations =================\n')
        sys.stdout.write('\t\tnumber of clusters: %d\n' % config['K'])
        sys.stdout.write('\t\tnumber of runs: %d\n' % config['numRun'])
        sys.stdout.write('\t\tmax number of iterations: %d\n' % config['max_iter'])
        sys.stdout.write('\t\tdistance ratio covar/img = %.4f\n' % config['r'])
        sys.stdout.write('\t\tdistance ratio set/img = %.4f\n' % config['rs'])
        sys.stdout.write('\t\tlambda1 = %.2f\tlambda2 = %.2f\n' % (config['lambda1'],config['lambda2']))
        sys.stdout.write('\t\ttransformation chosen: %s\n' % config['transform'])
        sys.stdout.write('\t\t=================================================\n')

    #============================ Preparing Data ======================================================
    # separate data into patient and normal groups
    feat_img = np.transpose(feat_img)
    x = feat_img[:, group == -1] # normal controls
    y = feat_img[:, group == 1] # patients
    xd = []
    yd = []
    xs = []
    ys = []
    if feat_cov is not None:
        feat_cov = np.transpose(feat_cov)
        xd = feat_cov[:, group == -1]
        yd = feat_cov[:, group == 1]
    if feat_set is not None:
        feat_set = np.transpose(feat_set)
        xs = feat_set[:, group == -1]
        ys = feat_set[:, group == 1]

    #================================Perform Clustering (2 modes available)=================================
    if config['mode'] == 'energy_min': #save result yields minimal energy
        obj = np.float('inf')
        for i in range(config['numRun']):
            cur_result = optimize(x, xd, xs, y, yd, ys, config)  #### optimize the ojectives
            cur_obj = cur_result[2].min()
            if config['verbose']:
                sys.stdout.write('\t\tRun id %d, obj = %f\n' % (i, cur_obj))
            else:
                time_bar(i, config['numRun'])
            if cur_obj < obj:
                result = cur_result
                obj = cur_obj
        sys.stdout.write('\n')
        membership = np.dot(result[1], Tr(result[0]['delta']))
        label = np.argmax(membership, axis=1)
    elif config['mode'] == 'reproducibility': # save result most reproducible
        label_mat = []
        results = []
        for i in range(config['numRun']):
            cur_result = optimize(x, xd, xs, y, yd, ys, config)
            membership = np.dot(cur_result[1], Tr(cur_result[0]['delta']))
            label = np.argmax(membership, axis=1)
            label_mat.append(label)
            results.append(cur_result)
            time_bar(i, config['numRun'])
        sys.stdout.write('\n')
        label_mat = np.asarray(label_mat)
        ari_mat = np.zeros((config['numRun'], config['numRun']))
        for i in range(config['numRun']):
            for j in range(i+1, config['numRun']):
                ari_mat[i, j] = ARI(label_mat[i, :], label_mat[j, :])
                ari_mat[j, i] = ari_mat[i, j]
        ave_ari = np.sum(ari_mat, axis=0)/(config['numRun']-1)
        idx = np.argmax(ave_ari)
        if config['verbose']: sys.stdout.write('\t\tBest average ARI is %f\n' % (max(ave_ari)))
        label = label_mat[idx, :]
        result = results[idx]

    #================================ Finalizing and Save =====================================
    sys.stdout.write('\tSaving results...\n')
    df_feature_pt = df_feature.loc[df_feature['diagnosis'].isin([1])]
    df_final = df_feature_pt[["participant_id", "session_id", "diagnosis"]]
    label = label + 1
    cluster_assignment_column = "assignment_" + str(config['K'])
    df_final[cluster_assignment_column] = label
    df_final.to_csv(outFile, index=False, sep='\t', encoding='utf-8')

    if config['modelFile']:
        output_model = os.path.join(output_dir, 'clustering_model.pkl')
        trainData = {'x': x, 'xd': xd, 'xs': xs, 'datasetID': unique_ID}
        model.update({'trainData': trainData})
        model.update({'model': result})
        model.update({'config': config})
        with open(output_model, 'wb') as f:
            pickle.dump(model, f, 2)
        
#==============================================================================================
# Optimization code
#==============================================================================================        
def optimize(x,xd,xs,y,yd,ys,config):
    """Expectation-Maximization optimization
    """
    params = initialization(x,y,config['K'])
    
    iteration = 0
    obj = np.zeros(config['max_iter']+1)
    obj[obj==0] = np.float('inf')
    
    while iteration < config['max_iter']:
        tx = transform(x,params)
        P = Estep(y,yd,ys,tx,xd,xs,params['sigsq'],config['r'],config['rs'])
        params = Mstep(y,yd,ys,x,tx,xd,xs,P,params,config)
    
        #> calculate objective function
        obj[iteration+1] = calc_obj(x,y,xd,yd,xs,ys,P,params,config)
        
        #> convergence
        if not config['quiet']: sys.stdout.write('obj = %f\n' % obj[iteration+1])
        if abs(obj[iteration] - obj[iteration+1]) < config['eps']:
            break
        iteration += 1
    
    # Save memory load by using float32
    params['T'] = params['T'].astype(np.float32)
    params['t'] = params['t'].astype(np.float32)
    params['delta'] = params['delta'].astype(np.float32)
    params['sigsq'] = params['sigsq'].astype(np.float32)
    P = P.astype(np.float32)
    return (params,P,obj)


def clustering_test(dataFile,outFile,modelFile):
    """Test function of CHIMERA
       Please be extremely careful when using this function.
       The ordering of normal controls should be exactly the same as training phase
    """
    #================================= Reading Data ======================================================
    sys.stdout.write('\treading model...\n')
    with open(modelFile) as f:
        model = pickle.load(f)
    trainData = model['trainData']
    params = model['model'][0]
    config = model['config']

    sys.stdout.write('\treading data...\n')
    feat_cov = None
    feat_set = None
    ID = None
    with open(dataFile) as f:
        data   = list(csv.reader(f))
        header = np.asarray(data[0])
        if 'IMG' not in header:
            sys.stdout.write('Error: image features not found. Please check csv header line for field "IMG".\n')
            sys.exit(1)
        data = np.asarray(data[1:])
        
        feat_img = (data[:,np.nonzero(header=='IMG')[0]]).astype(np.float)
        feat_cov = []
        feat_set = []
        if len(trainData['xd']) != 0:
            if 'COVAR' not in header:
                sys.stdout.write('Error: covariate features not found. Please check csv header line for field "COVAR".\n')
                sys.exit(1)
            feat_cov = (data[:,np.nonzero(header=='COVAR')[0]]).astype(np.float)
        if len(trainData['xs']) != 0:
            if 'Set' not in header:
                sys.stdout.write('Error: dataset ID not found. Please check csv header line for field "Set".\n')
                sys.exit(1)
            feat_set = data[:,np.nonzero(header=='Set')[0]].flatten()
            datasetID = np.copy(feat_set)
            feat_set  = np.zeros((len(datasetID),len(trainData['datasetID'])))
            for i in range(len(trainData['datasetID'])):
                feat_set[np.nonzero(datasetID==trainData['datasetID'][i])[0],i] = 1
        if 'ID' in header:
            ID = data[:,np.nonzero(header=='ID')[0]]
    
    #================================= Normalizing Data ======================================================
    if config['norm'] != 0 :
        feat_img, feat_cov = data_normalization_test(feat_img, feat_cov, model, config)
    
    #============================ Preparing Data ======================================================
    # separate data into patient and normal groups
    x  = trainData['x'] # normal controls
    y  = np.transpose(feat_img) # patients
    xd = trainData['xd']
    yd = np.transpose(feat_cov)
    xs = trainData['xs']
    ys = np.transpose(feat_set)
    
    #================================Perform Clustering ========================================
    sys.stdout.write('\tclustering...\n')
    tx = transform(x,params)
    P = Estep(y,yd,ys,tx,xd,xs,params['sigsq'],config['r'],config['rs'])
    membership = np.dot(P,Tr(params['delta']))
    label = np.argmax(membership, axis=1)
    
    #================================ Finalizing and Save =====================================
    sys.stdout.write('\tsaving results...\n')
    with open(outFile,'w') as f:
        if ID is None:
            f.write('Cluster\n')
            for i in range(len(label)):
                f.write('%d\n' % (label[i]+1))
        else:
            f.write('ID,Cluster\n')
            for i in range(len(label)):
                f.write('%s,%d\n' % (ID[i][0], label[i]+1))

#==============================================================================================
# Normalization code
#==============================================================================================  
def data_normalization(feat_img, feat_cov, config):
    if config['norm'] == 'minmax':
        model = {'img_min': 0, 'img_range': 0, 'cov_min': 0, 'cov_range': 0}
        model['img_min'] = feat_img.min(axis=0)
        feat_img = feat_img - feat_img.min(axis=0)
        model['img_range'] = feat_img.max(axis=0)
        feat_img = feat_img / feat_img.max(axis=0)
        if feat_cov is None:
            config['r'] = 0
        else:
            model['cov_min'] = feat_cov.min(axis=0)
            feat_cov = feat_cov - feat_cov.min(axis=0)          
            model['cov_range'] = feat_cov.max(axis=0)
            feat_cov = feat_cov / feat_cov.max(axis=0)
    elif config['norm'] == 'zscore':
        model = {'img_mean': 0, 'img_std': 1, 'cov_mean': 0, 'cov_std': 1}
        model['img_mean'] = feat_img.mean(axis=0)
        model['img_std'] = feat_img.std(axis=0)
        feat_img = (feat_img - model['img_mean'])/model['img_std']
        if feat_cov is None:
            config['r'] = 0
        else:
            model['cov_mean'] = feat_cov.mean(axis=0)
            model['cov_std']  = feat_cov.std(axis=0)
            feat_cov = (feat_cov - model['cov_mean'])/model['cov_std']
    else:
        raise Exception("Standqrdization method not implemented...")
    return model, feat_img, feat_cov

def data_normalization_test(feat_img, feat_cov, model, config):
    if config['norm'] == 1:
        feat_img = feat_img - model['img_min']
        feat_img = feat_img / model['img_range']
        if config['r'] != 0:
            feat_cov = feat_cov - model['cov_min']
            feat_cov = feat_cov / model['cov_range']
    else:
        feat_img = (feat_img - model['img_mean'])/model['img_std']
        if config['r'] != 0:
            feat_cov = (feat_cov - model['cov_mean'])/model['cov_std']
    return feat_img, feat_cov

#==============================================================================================
# Progress Bar
#==============================================================================================
def time_bar(i,num):
    sys.stdout.write('\r')
    progress = (i+1)*1.0/num
    prog_int = int(progress*50)
    sys.stdout.write('\t\t[%s%s] %.2f%%' % ('='*prog_int,' '*(50-prog_int), progress*100))
    sys.stdout.flush()
