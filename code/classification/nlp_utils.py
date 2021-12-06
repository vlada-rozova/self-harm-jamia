import pandas as pd
import numpy as np
import re
from time import time

import gensim
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix

from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, fbeta_score, make_scorer, auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

import matplotlib.pyplot as plt
import seaborn as sns

def find_pattern(df, col, pattern, context=True):
    tmp = df[df[col].str.contains(pattern)]
    print("Found {} rows.".format(tmp.shape[0]))
    for ind, row in tmp.iterrows():
        if context:
            print(ind, ": ", re.findall(".{10}" + pattern + ".{10}", row[col]))
        else:
            print(ind, ": ", re.findall(pattern, row[col]))
            

def preprocess(text):
    
    # Convert to lower case
    text = text.lower()
    
    # Remove "\x7f"
    pattern = re.compile(r"\x7f")
    text = pattern.sub(r" ", text)
    # "`/c" to "with"
    pattern = re.compile("`/c")
    text = pattern.sub(r" ", text)
    # Remove backslashes
    pattern = re.compile("\d\\\\\d")
    text = pattern.sub(r"/", text)
    pattern = re.compile("\\\\")
    text = pattern.sub(r" ", text)
    # Fix "patientexpect"
    pattern = re.compile(r"patientexpect")
    text = pattern.sub(r" patient expect ", text)
    
    # "l)" to "left"
    pattern = re.compile("(^|\W)l\)")
    text = pattern.sub(r"\1 left ", text)
    # "r)" to "right"
    pattern = re.compile("(^|\W)r\)")
    text = pattern.sub(r"\1 right ", text)
    # "@" to "at"
    pattern = re.compile("@")
    text = pattern.sub(r" at ", text)
    # "#" to "fractured" if not followed by number
    pattern = re.compile("#(?!\d)")
    text = pattern.sub(r" fracture ", text)
    # "+ve" to "positive"
    pattern = re.compile("\+ve(?![a-z])")
    text = pattern.sub(r" positive ", text)
    # "-ve" to "positive"
    pattern = re.compile("\-ve(?![a-z])")
    text = pattern.sub(r" negative ", text)
    # "co operative" and "co-operative" to "cooperative"
    pattern = re.compile("co\sop|co-op")
    text = pattern.sub(r"coop", text)
    # "r/ship" to relationship
    pattern = re.compile("r/ships?")
    text = pattern.sub(r" relationship ", text)
    # Remove "+" after digit
    pattern = re.compile("(\d)\+")
    text = pattern.sub(r"\1 ", text)
    
    # Remove parentheses
    pattern = re.compile("\((.*)\)[,\.]?")
    text = pattern.sub(r" , \1, ", text)
    # Remove curly brackets
    pattern = re.compile("\((.*)\)")
    text = pattern.sub(r" . \1. ", text)
    
    # 1. Replace "preg" by "pregnant"
    pattern = re.compile("preg$|preg\.?(\W)")
    text = pattern.sub(r" pregnant \1", text)
    
    # 2. Replace "reg" by "regular"
    pattern = re.compile("irreg$|irreg\.?(\W)")
    text = pattern.sub(r" irregular \1", text)
    pattern = re.compile("reg$|reg\.?(\W)")
    text = pattern.sub(r" regular \1", text)
    
    # 3. Normalise respiratory rate
    pattern = re.compile("([^a-z])rr(?![a-z])|resp\srate|resp\W?(?=\d)")
    text = pattern.sub(r"\1 rr ", text)
    
    # 4. Normalise oxygen saturation
    pattern = re.compile("sp\s?[o0]2|sp2|spo02|sa\s?[o0]2|sats?\W{0,3}(?=\d)")
    text = pattern.sub(r" sao2 ", text) 
    pattern = re.compile("([^a-z])sp\W{0,3}(?=[19])")
    text = pattern.sub(r"\1 sao2 ", text)
    
    # 5. Normilise temperature
    pattern = re.compile("([^a-z])t(emp)?\W{0,3}(?=[34]\d)")
    text = pattern.sub(r"\1 temp ", text)

    # 6. Normalise hours
    pattern = re.compile("([^a-z])hrs|([^a-z])hours")
    text = pattern.sub(r"\1 hours ", text)
     
    # 7. Normalise heart rate
    pattern = re.compile("([^a-z])hr(?![a-z])")
    text = pattern.sub(r"\1 hr ", text)
    
    # 8. Normalise GCS
    pattern = re.compile("gsc")
    text = pattern.sub(r"gcs", text)
    
    # 9. Normalise on arrival
    pattern = re.compile("o/a|on arrival|on assessment")
    text = pattern.sub(r" o/a ", text)

    # Add spaces around "bp"
    pattern = re.compile("([^a-z])bp(?![a-z])")
    text = pattern.sub(r"\1 bp ", text)
    
    # Add spaces around "bmp", "bsl", "gcs"
    pattern = re.compile("(bpm|bsl|gcs)")
    text = pattern.sub(r" \1 ", text)
    
    # Remove duplicated punctuation marks [-/+_,?.] and spaces
    pattern = re.compile("-{2,}")
    text = pattern.sub(r"-", text)
    pattern = re.compile("/{2,}")
    text = pattern.sub(r"/", text)
    pattern = re.compile("\+{2,}")
    text = pattern.sub(r"+", text)
    pattern = re.compile("_{2,}")
    text = pattern.sub(r"_", text)
    pattern = re.compile(",{2,}")
    text = pattern.sub(r",", text)  
    pattern = re.compile("\?{2,}")
    text = pattern.sub(r"?", text)
    pattern = re.compile("\.{2,}")
    text = pattern.sub(r".", text)
    pattern = re.compile("\s{2,}")
    text = pattern.sub(r" ", text)
    
    return text


def plot_sparse_matrix_sample(X):
    sns.heatmap(X.todense()[:, np.random.randint(0, X_train.shape[1], 100)]==0, 
                vmin=0, vmax=1, cbar=False, cmap="YlGnBu_r")
    plt.title('Sparse Matrix Sample');
    
    
def get_vectorizer(vectorizer_mode, params):
    if vectorizer_mode == "select features":
        return FeatureSelector(params)
    elif vectorizer_mode == "word2vec":
        model_path = "./models/rmh_cleaned_w2v_model.bin"
        return MeanEmbeddingVectorizer(model_path)
    elif vectorizer_mode == "doc2vec":
        model_path = "./models/rmh_cleaned_d2v_model.bin"
        return DocEmbeddingVectorizer(model_path)
    else:
        return TfidfVectorizer(analyzer=params['analyzer'], 
                               stop_words=stopwords.words('english'), 
                               token_pattern=r'\S+',
                               ngram_range=params['ngram_range'],
                               min_df=2, 
                               use_idf=params['use_idf'])
    
    
class MeanEmbeddingVectorizer(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.word2vec = gensim.models.Word2Vec.load(model_path)
        self.dim = self.word2vec.wv.vectors[0].shape[0]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        tokenized_X = [doc.split() for doc in X]
                    
        return np.array([
            np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in tokenized_X
        ])
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
    
    
class DocEmbeddingVectorizer(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.doc2vec = gensim.models.Doc2Vec.load(model_path)
        self.dim = self.doc2vec.wv.vectors[0].shape[0]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        tokenized_X = [doc.split() for doc in X]
        return np.array([
            self.doc2vec.infer_vector(words) 
            for words in tokenized_X
        ])
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
        

class FeatureSelector(object):
    def __init__(self, params):
        self.vectorizer = []
        self.analyzer = params['analyzer']
        self.ngram_range = params['ngram_range']
        self.use_idf = params['use_idf']
        self.mode = params['mode']
        self.thresh = params['thresh']
        self.df_features = pd.DataFrame()
        
    def fit(self, X, y):
        self.vectorizer = TfidfVectorizer(analyzer=self.analyzer, 
                                          stop_words=stopwords.words('english'), 
                                          token_pattern=r'\S+',
                                          ngram_range=self.ngram_range,
                                          min_df=2, 
                                          use_idf=self.use_idf
                                         )
        X_ = self.vectorizer.fit_transform(X)
        feature_names = self.vectorizer.get_feature_names()
        
        if self.mode == "select k best":
            self.df_features = select_k_best(X_, y,
                                             feature_names, 
                                             k=self.thresh, 
                                             verbose=False)
        if self.mode == "select by pvalue":
            self.df_features = select_by_pvalue(X_, y,
                                                feature_names, 
                                                alpha=self.thresh, 
                                                verbose=False)
                                          
        self.vectorizer.set_params(vocabulary=self.df_features.feature.unique())
        self.vectorizer.fit(X, y)
                                          
        return self
        
    def transform(self, X):
        X = self.vectorizer.transform(X)
        return X
                                          
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X) 


def select_k_best(X, y, feature_names, k=5000, verbose=False):
    assert k > 1
    ch2 = SelectKBest(chi2, k=k)
    ch2.fit(X, y)
    
    df_features = pd.DataFrame({"feature" : np.asarray(feature_names)[ch2.get_support()], 
                                "score" : ch2.scores_[ch2.get_support()]})
    if verbose:
        print("Extracting %d best features by a chi-squared test..." % k)
        print("n_samples: {}, n_features: {}".format(X.shape[0], df_features.feature.nunique()))
        print()
        print("Selected features:", len(df_features))
        print("Top features:", ", ".join(df_features.sort_values(by="score").feature[:20]))
        
    return df_features


def select_by_pvalue(X, y, feature_names, alpha=0.05, verbose=False):
    assert alpha < 1
    df_features = pd.DataFrame()
    if y.max() > 1:
        for cat in np.unique(y):
            _, p = chi2(X, y==cat)
            df_features = df_features.append(pd.DataFrame({"feature" : feature_names, 
                                                           "p_value" : p, 
                                                           "y" : cat}))
            df_features = df_features[df_features["p_value"] < alpha]

        if verbose:
            print("Extracting features by a chi-squared test with p-value < %0.2f..." % alpha)
            print("n_samples: {}, n_features: {}".format(X.shape[0], df_features.feature.nunique()))
            print()
            for cat in np.unique(y):
                print("# {}:".format(cat))
                print("Selected features:", len(df_features[df_features.y == cat]))
                print("Top features:", ", ".join(df_features.loc[df_features.y == cat].sort_values(by="p_value").feature[:20]))
                print()
    else:
        _, p = chi2(X, y)
        df_features = df_features.append(pd.DataFrame({"feature" : feature_names, 
                                                       "p_value" : p, 
                                                       "y" : 1}))
        df_features = df_features[df_features["p_value"] < alpha]
        print("Extracting features by a chi-squared test with p-value < %0.2f..." % alpha)
        print("Selected features:", df_features.shape[0])
        print()
        
    return df_features


def reduce_dimension(X, method="svd", n_components=100):
    print("Performing dimensionality reduction using LSA...")
    if method == "nmf":
        lsa = NMF(n_components)
    else:
        lsa = TruncatedSVD(n_components)
    lsa.fit(X)
    print("n_samples: {}, n_features: {}".format(X.shape[0], n_components))
    print()
    
    if method == "svd":
        explained_variance = lsa.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
        print()
    return lsa


def average_pr_auc(y, y_proba):
    y_dummy = pd.get_dummies(y, drop_first=False).values
    pr_auc = []
    for i in range(y_dummy.shape[1]):
        prec, rec, _ = precision_recall_curve(y_dummy[:,i], y_proba[:,i])
        pr_auc.append(auc(rec, prec))
    
    return np.mean(pr_auc)


def benchmark_cv(clf, X, y, class_names):
    print('_' * 80)
    print()
    print("Model training: ")
    print(clf)
    
    t0 = time()
    
    y_proba = cross_val_predict(clf, X, y, cv=10, method="predict_proba")
    
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    return y_proba


def benchmark_cv_score(clf, X, y, class_names):
    print('_' * 80)
    print()
    print("Model training: ")
    print(clf)
    
    if len(class_names) == 2:
        scoring = {
                   "precision" : "precision",
                   "recall" : "recall",
                   "f1" : "f1", 
#                    "f2" : make_scorer(fbeta_score, beta=2, average='binary'), 
#                    "roc" : "roc_auc", 
                   "ap" : "average_precision"
                  }
        
    elif len(class_names) == 3:
        scoring = {"precision" : "precision_macro",
                   "recall" : "recall_macro",
                   "f1" : "f1_macro", 
                   "f2" : make_scorer(fbeta_score, beta=2, average='macro')}   
    
    t0 = time()
    
    scores = cross_validate(clf, X, y, n_jobs=-1, cv=10, scoring=scoring, return_estimator=True)
    
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    
    print("Average Precision: %0.3f (+/- %0.2f)" % (scores["test_precision"].mean(), 
                                                    scores["test_precision"].std() * 2))
    print("Average Recall: %0.3f (+/- %0.2f)" % (scores["test_recall"].mean(), 
                                                 scores["test_recall"].std() * 2))
    print("Average F1 score: %0.3f (+/- %0.2f)" % (scores["test_f1"].mean(), 
                                                   scores["test_f1"].std() * 2))
#     print("Average F2 score: %0.3f (+/- %0.2f)" % (scores["test_f2"].mean(), 
#                                                    scores["test_f2"].std() * 2))
#     print("Average ROC AUC score: %0.3f (+/- %0.2f)" % (scores["test_roc"].mean(), 
#                                                    scores["test_roc"].std() * 2))
    print("Average AP score: %0.3f (+/- %0.2f)" % (scores["test_ap"].mean(), 
                                                   scores["test_ap"].std() * 2))
    
    return scores
    
    
def grid_search_cv(clf, X, y, param_grid, scoring="f1_macro"):
    t0 = time()
    
    search = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=3, scoring=scoring)
    search_result = search.fit(X, y)
    
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    print("Best: %f using %s" % (search_result.best_score_, search_result.best_params_))
    means = search_result.cv_results_['mean_test_score']
    stds = search_result.cv_results_['std_test_score']
    params = search_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        

def random_search_cv(clf, X, y, param_distributions, scoring="f1_macro"):
    t0 = time()
    
    search = RandomizedSearchCV(estimator=clf, param_distributions=param_distributions, 
                                n_iter=100, n_jobs=-1, cv=3, scoring=scoring)
    search_result = search.fit(X, y)
    
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    print("Best: %f using %s" % (search_result.best_score_, search_result.best_params_))
    means = search_result.cv_results_['mean_test_score']
    stds = search_result.cv_results_['std_test_score']
    params = search_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

            
            
def evaluate_model(y, y_proba, class_names, string, thresh=None, show_plots=True, digits=2, save_figures=False, filename=""):
    n_outputs = y_proba.shape[1]
    if n_outputs > 2:
        average = 'macro'
    else:
        average = 'binary'
    
    # Generate predictions
    if thresh:
        y_pred = np.where(y_proba[:,-1] > thresh, 1, 0)
    else:
        if n_outputs == 1:
            y_pred = np.where(y_proba > 0.5, 1, 0)
        else:
            y_pred = np.argmax(y_proba, axis=1)
            
    print("Model evaluation on the %s set" % string)
    print()
    
    # Classification report
    print("Classification report:")
    print(classification_report(y, y_pred, digits=digits))
    
    # F2 score
    print("%s F2: %0.3f" % (average, fbeta_score(y, y_pred, average=average, beta=2)))

    # Average prediction score
    print("Average precision score: %0.3f" % average_precision_score(y, y_proba[:,-1]))  
    
    # Plot confusion matrix
    plt.figure();
    plt.rcParams['figure.figsize'] = (7, 5)
    sns.heatmap(confusion_matrix(y, y_pred, normalize="true"), 
                annot=confusion_matrix(y, y_pred),
                annot_kws={'fontsize' : 16}, fmt="d",
                cmap="Blues", cbar=False, 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix");
    
    if save_figures:
        plt.savefig(filename + "_CM.png", bbox_inches='tight', dpi=300, pad_inches=0);
    
    if show_plots:
        if n_outputs > 1:
            y_dummy = pd.get_dummies(y, drop_first=False).values
            
        # Plot ROC curves
        plt.figure();
        sns.lineplot(x=[0, 1], y=[0, 1], color=sns.color_palette()[0], lw=2, linestyle='--', label="Chance")
        if len(class_names) == 2:
            fpr, tpr, _ = roc_curve(y, y_proba[:,-1])
            roc_auc = roc_auc_score(y, y_proba[:,-1])
            sns.lineplot(x=fpr, y=tpr, lw=3, color=sns.color_palette()[1], 
                         label="AUC = %0.2f" % roc_auc)
        else:
            for i in range(n_outputs):
                fpr, tpr, _ = roc_curve(y_dummy[:,i], y_proba[:,i])
                roc_auc = roc_auc_score(y_dummy[:,i], y_proba[:,i], multi_class="ovr")
                sns.lineplot(x=fpr, y=tpr, lw=3, color=sns.color_palette()[1 + i], 
                             label=class_names[i] + " (AUC = %0.2f)" % roc_auc)

        plt.xlim([-0.01, 1.0])
        plt.ylim([-0.01, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend(loc="lower right", fontsize=16);
        
        if save_figures:
            plt.savefig(filename + "_ROC.png", bbox_inches='tight', dpi=300, pad_inches=0);

        # Plot precision-recall curves
        plt.figure();
        
#         if n_outputs == 1:
#             prec, rec, _ = precision_recall_curve(y, y_proba)
#             pr_auc = auc(rec, prec)
#             sns.lineplot(x=rec, y=prec, lw=3, color=sns.color_palette()[1], 
#                          label="AUC = %0.2f" % pr_auc)
#         else:
        colors = (sns.color_palette()[0], sns.color_palette()[3])
        for i in range(n_outputs):
            prec, rec, _ = precision_recall_curve(y_dummy[:,i], y_proba[:,i])
            pr_auc = auc(rec, prec)
            sns.lineplot(x=rec, y=prec, lw=3, color=colors[i], 
                         label=class_names[i] + " (AUC = %0.2f)" % pr_auc)

        plt.xlim([-0.01, 1.0])
        plt.ylim([-0.01, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.legend(loc="lower right", fontsize=16);
        
        if save_figures:
            plt.savefig(filename + "_PR.png", bbox_inches='tight', dpi=300, pad_inches=0);  
            
    return y_pred