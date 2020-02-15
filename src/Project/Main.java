package Project;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.text.DecimalFormat;

public class Main {
    private static final Logger LOGGER = LogManager.getLogger(Main.class.getSimpleName());

    public static void main(String[] args) throws Exception {
        Options options = new Options();
        Option oh = new Option("h", "help", false, "Show this help message");
        oh.setRequired(false);
        Option og = new Option("g", "gui", false, "Launch the GUI version of this program");
        og.setRequired(false);
        Option ob = new Option("b", "bayesian", false, "Use bayesian to predict or not");
        ob.setRequired(false);
        Option ot = new Option("t", "train-file", true, "The name of the training file");
        ot.setRequired(true);
        Option op = new Option("p", "predict-file", true, "The name of the predict data file");
        op.setRequired(false);
        Option ow = new Option("w", "window-size", true, "The window size for joining consecutive rows");
        ow.setRequired(true);
        Option ol = new Option("l", "ttl", true, "time to live");
        ol.setRequired(true);
        Option or = new Option("r", "top-gain-ratio", true, "topGainRatio");
        or.setRequired(true);
        Option oc = new Option("c", "number-of-cores", true, "The number of cores to run (default all cores)");
        oc.setRequired(false);
        options.addOption(oh).addOption(og).addOption(ob).addOption(ot).addOption(op).addOption(ow).addOption(ol).addOption(or).addOption(oc);

        DecisionTree trainDecisionTree = new DecisionTree();
        DecisionTree predictDecisionTree = new DecisionTree();

        CommandLineParser parser = new BasicParser();
        try {
            LOGGER.info("Parsing the args: {}", args);
            CommandLine cmd = parser.parse(options, args);

            if (cmd.hasOption('h')) {
                new HelpFormatter().printHelp(Main.class.getSimpleName(), options);
                return;
            }

            if (cmd.hasOption('g')) {
                GUI.launch();
                return;
            }

            if (cmd.hasOption('b')) {
                DecisionTree.bayesian = true;
            }

            trainDecisionTree.fileName = cmd.getOptionValue("t");

            if (cmd.hasOption("p")) {
                predictDecisionTree.fileName = cmd.getOptionValue("p");
            }

            DecisionTree.window = Integer.valueOf(cmd.getOptionValue("w"));
            DecisionTree.ttl = Integer.valueOf(cmd.getOptionValue("l"));
            TreeNode.topGainRatio = Double.valueOf(cmd.getOptionValue("r"));

            if (cmd.hasOption("c")) {
                TreeNode.cores = Integer.valueOf(cmd.getOptionValue("c"));
            }
        } catch (ParseException e) {
            LOGGER.error("Failed to parse command line properties: {}", e.getMessage());
            new HelpFormatter().printHelp(Main.class.getSimpleName(), options);
            System.exit(1);
        }

        LOGGER.info("Loading training data...");
        trainDecisionTree.readData_new();

        long startTime = System.currentTimeMillis();
        TreeNode root = new TreeNode(trainDecisionTree.Data, trainDecisionTree.attributes);

        LOGGER.debug("------------ TREE STARTED ------------\n" +
                root.printTree(""));
        LOGGER.debug("------------  TREE ENDED -------------");
        LOGGER.info("It took {} seconds to train the tree.", (System.currentTimeMillis() - startTime) / 1000.0);
        LOGGER.info("The ROOT tree accuracy is {}", root.getAccuracy());

        LOGGER.info("Predicting training data...");
        startTime = System.currentTimeMillis();
        double trainAccuracy = trainDecisionTree.predictDataset(root);
        LOGGER.info("The accuracy of training data is {}%", new DecimalFormat("#.00").format(trainAccuracy));
        LOGGER.info("It took {} seconds to predict training data.", (System.currentTimeMillis() - startTime) / 1000.0);

        if (predictDecisionTree.fileName != null) {
            LOGGER.info("Loading test predict data...");
            predictDecisionTree.readData_new();
            LOGGER.info("Predicting test predict data...");
            startTime = System.currentTimeMillis();
            double predictAccuracy = predictDecisionTree.predictDataset(root);
            LOGGER.info("The accuracy of test predict data is {}%", new DecimalFormat("#.00").format(predictAccuracy));
            LOGGER.info("It took {} seconds to predict real data.", (System.currentTimeMillis() - startTime) / 1000.0);
        }
    }
}
