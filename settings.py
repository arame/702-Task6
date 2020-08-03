import datetime
import os
import sys
class Settings:
    sample_size_cut = 0.3   # 30% of images for a person will be neutral, and 30% will be the emotion
    pathOutput = "../output/"
    pathInputFile = "../files/ck_final.pickle"
    pathSaveNet = "../save/"
    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    dateString = ""
    num_epochs = 60
    batch_size = 10 
    batch_norm = True
    learning_rate = 0.005

    momentum = 0.7 # No neededed for Adam optimizer
    weight_decay =0.001  # -L2 weight decay and dropout cannot be run at the same time (usually 0.0001)
    dropout = False
    drop_prob1 = 0.5
    drop_prob2 = 0.5
    optimizer = "Adam"  # Select "Adam" or "SGD"

    ################
    #Learning params 
    ################
    # step learning rate adjustment
    step_size = 20
    gamma = 0.5 
    # plateu learning rate adjustment
    factor = 0.5 
    patience = 2

    @staticmethod
    def start():
        print("*"*100)
        print("** Start at ", datetime.datetime.now())
        print("*"*100)
        Settings.dateString = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
        Settings.validateHyperparameters()
        Settings.pathOutput = Settings.setSettingPath(Settings.pathOutput)
        Settings.pathSaveNet = Settings.setSettingPath(Settings.pathSaveNet)
        Settings.printHyperparameters()

    @staticmethod
    def end():
        Settings.printHyperparameters()
        print("Output files are located in the folder ", Settings.pathOutput)
        print("*"*100)
        print("** End at ", datetime.datetime.now())
        print("*"*100)

    @staticmethod
    def validateHyperparameters():
        if Settings.weight_decay > 0 and Settings.dropout == True:
            sys.exit("!! Cannot run both l2 and dropout at the same time")

        if Settings.optimizer != "SGD" and Settings.optimizer != "Adam":
            sys.exit("!! Not a valid optimizer choice, select from 'SGD' or 'Adam'")

        if Settings.optimizer == "adam" and Settings.momentum != 0:
            sys.exit("!! Cannot use the adam optimizer with momentum")

    @staticmethod
    # Create a unique folder for the output files to avoid file name clashes and 
    # make it easier to locate output files for each run
    def setSettingPath(path):
        path = path + Settings.dateString
        os.makedirs(path)
        path = path + "/"
        return path

    @staticmethod
    def printHyperparameters():
        print("*"*100)
        print("* Hyperparameters")
        print("* ---------------")
        print("Learning Rate  ", Settings.learning_rate)
        print("Batch Size     ", Settings.batch_size)
        print("Epochs         ", Settings.num_epochs)
        print("Momentum       ", Settings.momentum)
        if Settings.drop_prob1 == 0 and Settings.drop_prob2 == 0:
            print("!! No Drop out")
        else:
            print("Dropout Rates = ", Settings.drop_prob1, " and ", Settings.drop_prob2)
        if Settings.weight_decay == 0:
            print("!! No L2 weight decay")
        else:
            print("L2 weight decay", Settings.weight_decay)

        if Settings.optimizer ==  "Adam":
            print("This is using the Adam Optimiser")
        else:
            print("This is using the SGD Optimiser")