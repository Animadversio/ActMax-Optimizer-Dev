tmpsavedir = r"E:\Cluster_Backup\Evol_tmp"


def test_Evolution(explabel, model_unit):
    from core.insilico_exps import ExperimentEvolution
    Exp = ExperimentEvolution(model_unit, savedir=tmpsavedir, explabel=explabel, )
    Exp.run()
    Exp.visualize_best()
    Exp.visualize_trajectory()
    Exp.save_last_gen()


def test_densenet121():
    explabel, model_unit = "densenet_fc1", ("densenet121", ".Linearclassifier", 1)
    test_Evolution(explabel, model_unit)


def test_densenet169():
    explabel, model_unit = "densenet_fc1", ("densenet169", ".Linearclassifier", 1)
    test_Evolution(explabel, model_unit)


def test_resnet50_linf8():
    explabel, model_unit = "resnet50_linf8_fc_01", ("resnet50_linf8", ".Linearfc", 1)
    test_Evolution(explabel, model_unit)


def test_alexnet():
    # explabel, model_unit = "alexnet_fc8_1", ("alexnet", "fc8", 1)
    explabel, model_unit = "alexnet_fc8_1", ("alexnet", ".classifier.Linear6", 1)
    test_Evolution(explabel, model_unit)


def test_vgg16():
    explabel, model_unit = "alexnet_fc8_1", ("alexnet", ".classifier.Linear6", 1)
    test_Evolution(explabel, model_unit)