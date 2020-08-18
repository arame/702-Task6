import datetime
import os
import sys
class Settings:
    pathOutput = "../output/"
    pathInputFile = "../files/ck_final.pickle"
    pathSaveNet = "../save/"
    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    dateString = ""
    num_epochs = 350
    batch_size = 10 
    batch_norm = True
    learning_rate = 0.005
    isPlateau = True
    momentum = 0            # Not neededed for Adam optimizer
    weight_decay =0     # L2 weight decay and dropout cannot be run at the same time (usually 0.0001)
    dropout = False
    drop_prob1 = 0.5
    drop_prob2 = 0.5
    optimizer = "Adam"      # Select "Adam" or "SGD"

    ################
    #Learning params 
    ################
    # step learning rate adjustment
    '''Decays the learning rate of each parameter group by gamma every step_size epochs. 
    Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler. 
    When last_epoch=-1, sets initial lr as lr.
    '''
    step_size = 20
    gamma = 0.5 
    # plateu learning rate adjustment
    '''Reduce learning rate when a metric has stopped improving. 
    Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. 
    This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’ number of epochs, the learning rate is reduced.
    '''
    factor = 0.5            # Factor by which the learning rate will be reduced. new_lr = lr * factor.
    patience = 2            # Number of epochs with no improvement after which learning rate will be reduced. 
                            # For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, 
                            # and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then.

    @staticmethod
    def start():
        print("*"*100)
        print("** Start at ", datetime.datetime.now())
        print("*"*100)
        Settings.dateString = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
        Settings.validateHyperparameters()
        if not os.path.exists(Settings.pathOutput):
            os.makedirs(Settings.pathOutput)
        if not os.path.exists(Settings.pathSaveNet):
            os.makedirs(Settings.pathSaveNet)
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
        filepath = Settings.pathOutput + "hyperparameters.txt"
        file = open(filepath, "w+")
        Settings.outputLine(file, "* Hyperparameters")
        Settings.outputLine(file, "* ---------------")
        Settings.outputLine(file, "Learning Rate  " + str(Settings.learning_rate))
        Settings.outputLine(file, "Batch Size     " + str(Settings.batch_size))
        Settings.outputLine(file, "Epochs         " + str(Settings.num_epochs))
        Settings.outputLine(file, "Momentum       " + str(Settings.momentum))
        if Settings.drop_prob1 == 0 and Settings.drop_prob2 == 0:
            Settings.outputLine(file, "!! No Drop out")
        else:
            Settings.outputLine(file, "Dropout Rates = " + str(Settings.drop_prob1) + " and " + str(Settings.drop_prob2))
        if Settings.weight_decay == 0:
            Settings.outputLine(file, "!! No L2 weight decay")
        else:
            Settings.outputLine(file, "L2 weight decay" + str(Settings.weight_decay))

        if Settings.optimizer ==  "Adam":
            Settings.outputLine(file, "This is using the Adam Optimiser")
        else:
            Settings.outputLine(file, "This is using the SGD Optimiser")

        if Settings.isPlateau:
            Settings.outputLine(file, "Optimizer learning rate plateau settings - Patience; " + str(Settings.patience) + " Factor; " + str(Settings.factor))
        else:
            Settings.outputLine(file, "Optimizer learning rate step settings - Step Size; " + str(Settings.step_size) + " Gamma; " + str(Settings.gamma))

        Settings.outputLine(file, "Output files are located in the folder " + Settings.pathOutput)

    @staticmethod
    def outputLine(file, message):
        file.write(message + "\n")
        print(message)