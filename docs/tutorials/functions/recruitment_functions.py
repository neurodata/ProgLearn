import numpy as np
from math import log2, ceil 
import random

from proglearn.progressive_learner import ClassificationProgressiveLearner
from proglearn.transformers import TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter
from proglearn.deciders import SimpleArgmaxAverage


class PosteriorsByTree(SimpleArgmaxAverage):
    """Variation on the decider class SimpleArgmaxAverage to allow for return of posterior probabilities by tree.
    """
    
    def predict_proba(self, X, transformer_ids=None):
        """
        """
        #vote_per_transformer_id = []
        vote_per_alltrees = []
        for transformer_id in (
            transformer_ids
            if transformer_ids is not None
            else self.transformer_id_to_voters.keys()
        ):
            if not self.is_fitted():
                msg = (
                    "This %(name)s instance is not fitted yet. Call 'fit' with "
                    "appropriate arguments before using this decider."
                )
                raise NotFittedError(msg % {"name": type(self).__name__})

            #vote_per_bag_id = []
            for bag_id in range(len(self.transformer_id_to_transformers[transformer_id])):
                transformer = self.transformer_id_to_transformers[transformer_id][bag_id]
                X_transformed = transformer.transform(X)
                voter = self.transformer_id_to_voters[transformer_id][bag_id]
                vote = voter.predict_proba(X_transformed)
                #vote_per_bag_id.append(vote) #posteriors per tree: 50 x (37,10)
                vote_per_alltrees.append(vote)
            #vote_per_transformer_id.append(np.mean(vote_per_bag_id, axis=0)) #avg over each bag (forest)
            #saa = np.mean(vote_per_transformer_id, axis=0) # original simpleargmaxaverage output
        return vote_per_alltrees
    
    def predict(self, X, transformer_ids=None):
        """
        """
        vote_per_alltrees = self.predict_proba(X, transformer_ids=transformer_ids)
        vote_overall = np.mean(vote_per_alltrees, axis=0)
        return self.classes[np.argmax(vote_overall, axis=1)]


def sort_data(data_x, data_y, num_points_per_task, total_task=10, shift=1):
    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]
    train_x_across_task = []
    train_y_across_task = []
    test_x_across_task = []
    test_y_across_task = []

    batch_per_task=5000//num_points_per_task
    sample_per_class = num_points_per_task//total_task
    test_data_slot=100//batch_per_task

    for task in range(total_task):
        for batch in range(batch_per_task):
            for class_no in range(task*10,(task+1)*10,1):
                indx = np.roll(idx[class_no],(shift-1)*100)
                
                if batch==0 and class_no==task*10:
                    train_x = x[indx[batch*sample_per_class:(batch+1)*sample_per_class],:]
                    train_y = y[indx[batch*sample_per_class:(batch+1)*sample_per_class]]
                    test_x = x[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500],:]
                    test_y = y[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500]]
                else:
                    train_x = np.concatenate((train_x, x[indx[batch*sample_per_class:(batch+1)*sample_per_class],:]), axis=0)
                    train_y = np.concatenate((train_y, y[indx[batch*sample_per_class:(batch+1)*sample_per_class]]), axis=0)
                    test_x = np.concatenate((test_x, x[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500],:]), axis=0)
                    test_y = np.concatenate((test_y, y[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500]]), axis=0)
        
        train_x_across_task.append(train_x)
        train_y_across_task.append(train_y)
        test_x_across_task.append(test_x)
        test_y_across_task.append(test_y)

    return train_x_across_task, train_y_across_task, test_x_across_task, test_y_across_task


def experiment(data_x, data_y, ntrees, reps, estimation_set, num_points_per_task, num_points_per_forest, task_10_sample):
    
    # sort data
    train_x_across_task, train_y_across_task, test_x_across_task, test_y_across_task = sort_data(data_x,data_y,num_points_per_task)
    
    # create matrices for storing values
    hybrid = np.zeros(reps,dtype=float)
    building = np.zeros(reps,dtype=float)
    recruiting= np.zeros(reps,dtype=float)
    uf = np.zeros(reps,dtype=float)
    mean_accuracy_dict = {'hybrid':[],'building':[],'recruiting':[],'UF':[]}
    std_accuracy_dict = {'hybrid':[],'building':[],'recruiting':[],'UF':[]}

    # iterate over all sample sizes ns
    for ns in task_10_sample: 

        # size of estimation and validation sample sets
        estimation_sample_no = ceil(estimation_set*ns)
        validation_sample_no = ns - estimation_sample_no

        # repeat `rep` times
        for rep in range(reps):
            print("doing {} samples for {} th rep".format(ns,rep))

            # initiate lifelong learner
            l2f = ClassificationProgressiveLearner(
                default_transformer_class=TreeClassificationTransformer,
                default_transformer_kwargs={},
                default_voter_class=TreeClassificationVoter,
                default_voter_kwargs={
                    "finite_sample_correction": False
                },
                #default_decider_class=SimpleArgmaxAverage,
                default_decider_class=PosteriorsByTree,
                default_decider_kwargs={},
            )

            # train l2f on first 9 tasks
            for task in range(9):
                indx = np.random.choice(num_points_per_task, num_points_per_forest, replace=False)
                cur_X = train_x_across_task[task][indx]
                cur_y = train_y_across_task[task][indx]

                # if task number is 0, add task; else, add transformer for task
                if task == 0:
                    l2f.add_task(
                        cur_X, 
                        cur_y,
                        num_transformers = ntrees,
                        transformer_kwargs={"kwargs":{"max_depth": ceil(log2(num_points_per_forest))}},
                        voter_kwargs={"classes": np.unique(cur_y),"finite_sample_correction": False},
                        decider_kwargs={"classes": np.unique(cur_y)}
                    )
                else:
                    l2f.add_transformer(
                        cur_X, 
                        cur_y,
                        num_transformers = ntrees,
                        transformer_kwargs={"kwargs":{"max_depth": ceil(log2(estimation_sample_no))}},
                        voter_kwargs={"classes": np.unique(cur_y),"finite_sample_correction": False},
                        #decider_kwargs={"classes": np.unique(cur_y)}
                    )

            # train l2f on 10th task
            task_10_train_indx = np.random.choice(num_points_per_task, ns, replace=False)
            cur_X = train_x_across_task[9][task_10_train_indx[:estimation_sample_no]]
            cur_y = train_y_across_task[9][task_10_train_indx[:estimation_sample_no]]
            l2f.add_transformer(
                cur_X, 
                cur_y,
                num_transformers = ntrees,
                transformer_kwargs={"kwargs":{"max_depth": ceil(log2(estimation_sample_no))}},
                voter_kwargs={"classes": np.unique(cur_y),"finite_sample_correction": False},
                #decider_kwargs={"classes": np.unique(cur_y)}
            )

            ## L2F validation
            # get posteriors for l2f on first 9 tasks
            # want posteriors_across_trees to have dimension 9*ntrees, validation_sample_no, 10
            posteriors_across_trees = l2f.predict_proba(
                train_x_across_task[9][task_10_train_indx[estimation_sample_no:]],
                task_id=0,
                transformer_ids=[0,1,2,3,4,5,6,7,8]
                )
            if len(posteriors_across_trees) != 9*ntrees: ############################################
                print("ERROR IN NUMBER OF TREES")
            # compare error in each tree and choose best 25/50 trees
            error_across_trees = np.zeros(9*ntrees)
            validation_target = train_y_across_task[9][task_10_train_indx[estimation_sample_no:]]
            for tree in range(len(posteriors_across_trees)):
                res = np.argmax(posteriors_across_trees[tree],axis=1) + 90
                error_across_trees[tree] = 1-np.mean(validation_target==res)
            best_50_tree = np.argsort(error_across_trees)[:50]
            best_25_tree = best_50_tree[:25]

            ## uf trees validation
            # get posteriors for l2f on only the 10th task
            posteriors_across_trees = l2f.predict_proba(
                train_x_across_task[9][task_10_train_indx[estimation_sample_no:]],
                task_id=0,
                transformer_ids=[9]
                )
            # compare error in each tree and choose best 25 trees
            error_across_trees = np.zeros(ntrees)
            validation_target = train_y_across_task[9][task_10_train_indx[estimation_sample_no:]]
            for tree in range(ntrees):
                res = np.argmax(posteriors_across_trees[tree],axis=1) + 90
                error_across_trees[tree] = 1-np.mean(validation_target==res)
            best_25_uf_tree = np.argsort(error_across_trees)[:25]

            ## evaluation
            # get posteriors for all data points in test set using first 9 tasks
            posteriors_across_trees = l2f.predict_proba(
                test_x_across_task[9],
                task_id=0,
                transformer_ids=[0,1,2,3,4,5,6,7,8]
                )

            # train 10th tree under each scenario: building, recruiting, hybrid, UF
            # RECRUITING
            recruiting_posterior = np.mean(np.array(posteriors_across_trees)[best_50_tree],axis=0)
            res = np.argmax(recruiting_posterior,axis=1) + 90
            recruiting[rep] = 1 - np.mean(test_y_across_task[9]==res)
            # BUILDING
            building_res = l2f.predict(test_x_across_task[9],task_id=0,transformer_ids=[0,1,2,3,4,5,6,7,8,9]) + 90
            building[rep] = 1 - np.mean(test_y_across_task[9]==building_res)
            # UF
            uf_res = l2f.predict(test_x_across_task[9],task_id=0,transformer_ids=[9]) + 90
            uf[rep] = 1 - np.mean(test_y_across_task[9]==uf_res)
            # HYBRID
            posteriors_across_trees_hybrid_uf = l2f.predict_proba(test_x_across_task[9],task_id=0,transformer_ids=[9])
            hybrid_posterior_all = np.concatenate((np.array(posteriors_across_trees)[best_25_tree],np.array(posteriors_across_trees_hybrid_uf)[best_25_uf_tree]),axis=0)
            hybrid_posterior = np.mean(hybrid_posterior_all,axis=0)
            hybrid_res = np.argmax(hybrid_posterior,axis=1) + 90
            hybrid[rep] = 1 - np.mean(test_y_across_task[9]==hybrid_res)

        # calculate mean and stdev for each
        mean_accuracy_dict['hybrid'].append(np.mean(hybrid))
        std_accuracy_dict['hybrid'].append(np.std(hybrid,ddof=1))
        mean_accuracy_dict['building'].append(np.mean(building))
        std_accuracy_dict['building'].append(np.std(building,ddof=1))
        mean_accuracy_dict['recruiting'].append(np.mean(recruiting))
        std_accuracy_dict['recruiting'].append(np.std(recruiting,ddof=1))
        mean_accuracy_dict['UF'].append(np.mean(uf))
        std_accuracy_dict['UF'].append(np.std(uf,ddof=1))
        
    return mean_accuracy_dict, std_accuracy_dict