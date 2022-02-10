""" Run optimization with nevergrad interface
Record and plot the score traces of nevergrad optimizers
"""
import nevergrad as ng
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.optimization.callbacks import ParametersLogger
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import pickle as pkl
from os.path import join
figdir = r"E:\OneDrive - Harvard University\GECCO2022\Figures\NGBenchmark"
mpl.rcParams['pdf.fonttype'] = 42 # set font for export to pdfs
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

class LossTraceLogger:
    """Adapted from the official ParametersLogger, just log the loss and some hyperparameter
    Logs parameter and run information throughout into a file during
    optimization.
    Parameters
    ----------
    filepath: str or pathlib.Path
        the path to dump data to
    append: bool
        whether to append the file (otherwise it replaces it)
    order: int
        order of the internal/model parameters to extract
    Example
    -------
    .. code-block:: python
        logger = ParametersLogger(filepath)
        optimizer.register_callback("tell",  logger)
        optimizer.minimize()
        list_of_dict_of_data = logger.load()
    Note
    ----
    Arrays are converted to lists
    """

    def __init__(self, filepath: tp.Union[str, Path], append: bool = True, order: int = 1) -> None:
        # self._session = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
        self._filepath = Path(filepath)
        # self._order = order
        # if self._filepath.exists() and not append:
        #     self._filepath.unlink()  # missing_ok argument added in python 3.8
        self._filepath.parent.mkdir(exist_ok=True, parents=True)
        self.data_col = []

    def __call__(self, optimizer, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        data = {
            # "#parametrization": optimizer.parametrization.name,
            "#optimizer": optimizer.name,
            "#num-ask": optimizer.num_ask,
            "#num-tell": optimizer.num_tell,
            "#num-tell-not-asked": optimizer.num_tell_not_asked,
            "#generation": candidate.generation,
            "#loss": loss,
        }
        self.data_col.append(data)

    def dump(self):
        pkl.dump(self.data_col, open(self._filepath,"wb"), )

    def save_csv(self, path):
        df = pd.DataFrame(self.data_col)
        df.to_csv(path)
        return df

    def loss_arr(self) -> tp.List[tp.Dict[str, tp.Any]]:
        """Loads data from the log file"""
        losses = np.array([d["#loss"] for d in self.data_col])
        return losses

#%%
from core.GAN_utils import upconvGAN
from core.CNN_scorers import TorchScorer
G = upconvGAN("fc6")
G.eval().cuda()
scorer = TorchScorer("resnet50_linf8") #
# scorer.select_unit((None,'.layer1.Bottleneck2',1,28,28))
# scorer.select_unit((None,'.layer2.Bottleneck3',1,14,14))
# scorer.select_unit((None,'.layer3.Bottleneck5',1,7,7))
# scorer.select_unit((None,'.layer4.Bottleneck2',1,4,4))
scorer.select_unit((None,'.Linearfc',1))
def score_batch(z, ):
    return -scorer.score(G.visualize_batch_np(z.reshape([-1,4096])))

instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)).set_bounds(-6, 6),)
#%%
import matplotlib.pylab as plt
import seaborn as sns
optimlist = ["NGOpt", "DE", "TwoPointsDE",
             "ES", "CMA", "RescaledCMA", "DiagonalCMA", "SQPCMA", #'RealSpacePSO',
             "PSO", "OnePlusOne", "TBPSA",
             "RandomSearch"]
loss_arr_col = {}
for optimnm in optimlist:
    optimizer = ng.optimizers.registry[optimnm](parametrization=instrum, budget=3000, num_workers=40,)
    logger = LossTraceLogger(filepath=r"ng_logger/%s_optimlog.pkl"%optimnm)
    logger_all = ParametersLogger(filepath=r"ng_logger/%s_optimlog_all.json"%optimnm)
    optimizer.register_callback("tell",  logger)
    optimizer.register_callback("tell",  logger_all)
    optimizer.minimize(score_batch, verbosity=True, batch_mode=True)
    logger.dump()
    df = logger.save_csv(r"ng_logger/%s_optimlog.csv"%optimnm)
    loss_arr_col[optimnm] = logger.loss_arr()

    plt.figure()
    sns.lineplot(x=df["#generation"], y=-df["#loss"])
    plt.savefig(r"ng_logger/%s_optim_trace.png"%optimnm)
    plt.show()
#%%
df_all = pd.DataFrame()
for optim in loss_arr_col:
    df = pd.read_csv(r"ng_logger/%s_optimlog.csv"%optim)
    df_all[optim] = - df["#loss"]

#%% max value plot
def plot_traj_cmp(df_all, optimorder=None, scatter=True):
    if optimorder is None:
        optimorder = list(df_all)
    plt.figure(figsize=[8, 8])
    for optim in optimorder:
        if scatter:
            plt.scatter(np.arange(1,3001), df_all[optim], s=4, alpha=0.1, label=None)
        plt.plot(np.arange(1,3001), df_all[optim].cummax(), lw=2, alpha=1, label=optim)
    plt.xlabel("Function Evaluation", fontsize=16)
    plt.ylabel("Activation", fontsize=16)
    plt.title("Score Traces Comparison\nResNet50-Robust Linear Unit 1", fontsize=18)
    plt.legend(fontsize=14)
    plt.savefig(join(figdir, r"example_score_traces%s.png" % ("_w_scatter" if scatter else "")))
    plt.savefig(join(figdir, r"example_score_traces%s.pdf" % ("_w_scatter" if scatter else "")))
    plt.show()

optimorder = ['CMA', 'DiagonalCMA', 'SQPCMA', 'RescaledCMA', 'ES',
             'NGOpt', 'DE', 'TwoPointsDE',
             'PSO', 'OnePlusOne', 'TBPSA',
             'RandomSearch']

plot_traj_cmp(df_all, optimorder=optimorder, scatter=True)
plot_traj_cmp(df_all, optimorder=optimorder, scatter=False)
#%%
def plot_traj_montage(df_all, optimorder=None, scatter=True, legend=True):
    if optimorder is None:
        optimorder = list(df_all)
    figh, axs = plt.subplots(3,4,figsize=[10, 8.5])
    for i, optim in enumerate(optimorder):
        df = pd.read_csv(r"ng_logger/%s_optimlog.csv" % optim)
        plt.sca(axs.flatten()[i],)
        if scatter:
            sns.scatterplot(x=df["#generation"], y=-df["#loss"],
                    s=12, alpha=0.15, **({"legend":None} if ~legend else {}))
        sns.lineplot(x=df["#generation"], y=-df["#loss"], label=optim,
                     color="red", **({"legend":None} if ~legend else {}))
        plt.title(optim)
        plt.xlabel(None)
        plt.ylabel(None)
    figh.suptitle("Score Traces Comparison\nResNet50-Robust Linear Unit 1", fontsize=16)
    plt.savefig(join(figdir, r"example_montage_score_traces_indiv%s.png" % ("_w_scatter" if scatter else "")))
    plt.savefig(join(figdir, r"example_montage_score_traces_indiv%s.pdf" % ("_w_scatter" if scatter else "")))
    plt.show()

plot_traj_montage(df_all, optimorder=optimorder, scatter=False, legend=False)
plot_traj_montage(df_all, optimorder=optimorder, scatter=True, legend=False)