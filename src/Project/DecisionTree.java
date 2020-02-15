package Project;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.StringTokenizer;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class DecisionTree {

    public static boolean classIncluded = false;
    public static boolean bayesian = false;
    public static int window;
    public static boolean randomly = true;
    public static boolean strict = true;
    public static boolean update = false;
    public static int ttl;

    public HashSet<Integer> nominals = new HashSet();
    public ArrayList<String[]> Data = new ArrayList();
    public Attribute[] attributes;
    public String[] orgAttNames;
    public String fileName = null;
    private static final Logger logger = LogManager.getLogger(DecisionTree.class.getSimpleName());

    /**
     * Predict this Data with the given decision tree
     *
     * @param root root node of the trained decision tree
     * @return prediction accuracy
     */
    public double predictDataset(TreeNode root) {

        int less = 0;
        int correctCount = 0;

        for (int i = 0; i < Data.size(); i++) {

            String[] record = Data.get(i);
            TreeNode leaf = root.predict(record);
            String classPredicted = null;

            if (leaf.getEntropy() == 0) {
                logger.trace("This is a 0 entropy leaf");
                classPredicted = TreeNode.Data.get(leaf.getRecord(0))[TreeNode.attributes.length - 1];
            } else if (!bayesian) {
                logger.trace("This is a non-bayesian leaf");
                int rnd = (int) (Math.random() * leaf.getRecordSize());
                classPredicted = TreeNode.Data.get(leaf.getRecord(rnd))[TreeNode.attributes.length - 1];
            } else {
                logger.trace("This is a bayesian leaf");
                int count = -Integer.MAX_VALUE;
                int c = 1;
                if (leaf.classCount == null)
                    logger.debug("LLLL");
                for (String val : leaf.classCount.keySet()) {
                    if (leaf.classCount.get(val) > count) {
                        count = leaf.classCount.get(val);
                        classPredicted = val;
                        c = 1;
                    } else if (leaf.classCount.get(val) == count) c++;
                }
                if (c != 1) {
                    if (randomly) {
                        int rnd = (int) (Math.random() * c);
                        int q = 0;
                        count = leaf.classCount.get(classPredicted);
                        for (String val : leaf.classCount.keySet()) {
                            if (leaf.classCount.get(val) == count) {
                                if (q == rnd) {
                                    classPredicted = val;
                                    break;
                                } else q++;
                            }
                        }
                    } else {
                        int mostCount = leaf.classCount.get(classPredicted);
                        for (String val : leaf.classCount.keySet()) {
                            if (leaf.classCount.get(val) == mostCount) {
                                classPredicted = "Wrong12345";
                                if (strict) break;
                                else {
                                    less++;
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if (classPredicted == null) {
                logger.fatal("Record No. {} in TestSet could not be predicted.", i);
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < record.length; j++)
                    sb.append(record[j] + ",");
                logger.error(sb.toString());
                logger.fatal("There might be a nominal attribute holding a value in testset that does not exist in trainset.");
                System.exit(1);
            }

            if (classPredicted.equals("Wrong12345")) continue;

            if (classPredicted.equals(record[attributes.length - 1])) {
                correctCount++;
                if (leaf.getEntropy() != 0 && update)
                    leaf.classCount.put(classPredicted, leaf.classCount.get(classPredicted) + 1);
            }
            // else {
            //     if(leaf.entropy!=0 && update)
            //       leaf.classCount.put(classPredicted, leaf.classCount.get(classPredicted)-1);
            // }
        }

        return 100 * ((double) correctCount / (double) (Data.size() - less));

    }

    /** Read data and merge
     *
     * The first column is user id, the second time
     *
     * @return
     * @throws Exception
     */
    public int readData_new() throws Exception {
        if (this.fileName == null)
            return 0;

        FileInputStream in = null;
        try {
            File inputFile = new File(this.fileName);
            in = new FileInputStream(inputFile);
        } catch (Exception e) {
            logger.fatal("Unable to open data file: {}", this.fileName);
            e.printStackTrace();
            return 0;
        }
        BufferedReader bin = new BufferedReader(new InputStreamReader(in));
        String input;
        while (true) {
            input = bin.readLine();
            if (input == null) {
                logger.fatal("No data found in the data file: {}", this.fileName);
                return 0;
            }
            if (input.startsWith("//")) continue;
            if (input.equals("")) continue;
            break;
        }
        logger.info("Reading input header: {}", input);
        StringTokenizer tokenizer = new StringTokenizer(input.trim(), ",");
        int orgNumAtt = tokenizer.countTokens() - 2;
        tokenizer.nextToken(); // move forward to skip entity id
        tokenizer.nextToken(); // move forward to skip time
        if (orgNumAtt <= 1) {
            logger.fatal("Read line: {}", input);
            logger.fatal("Could not obtain the names of attributes in the line\n" +
                    "Expecting at least one input attribute and one output attribute");
            return 0;
        }
        int numAttributes = classIncluded ? orgNumAtt * window : orgNumAtt * window - (window - 1);
        attributes = new Attribute[numAttributes];
        orgAttNames = new String[orgNumAtt];
        for (int i = 0; i < orgNumAtt; i++) {
            orgAttNames[i] = tokenizer.nextToken();
        }

        for (int i = 0; i < window; i++) {
            int org = orgNumAtt - (classIncluded ? 0 : 1);
            for (int j = 0; j < org; j++) {
                int h = org * i + j;
                attributes[h] = new Attribute(orgAttNames[j], i + 1, h, j == orgNumAtt - 1 ? true : nominals.contains(j));
            }
        }
        if (!classIncluded)
            attributes[numAttributes - 1] = new Attribute(orgAttNames[orgNumAtt - 1], window, numAttributes - 1, true);


        ArrayList<String[]> vecTok = new ArrayList(window);
        int count = 0;
        while (true) {
            count++;
            input = bin.readLine();
            if (input == null) break;
            if (input.startsWith("//") || input.startsWith("#")) continue;
            if (input.equals("")) continue;

            tokenizer = new StringTokenizer(input, ",");
            int numtokens = tokenizer.countTokens();
            if (numtokens != orgNumAtt + 2) {
                logger.fatal("Read {} data", count);
                logger.fatal("Last line read: {}", input);
                logger.fatal("Expecting {} attributes, but it is {}", orgNumAtt, numtokens);
                bin.close();
                return 0;
            }
            String[] orgRecord = new String[numtokens];
            for (int i = 0; i < numtokens; i++) {
                orgRecord[i] = tokenizer.nextToken();
            }

            // check the TTL and same entity
            if (!checkTTL(vecTok, orgRecord) || !checkEntity(vecTok, orgRecord)) {
                logger.trace("Entity {} time {} causes sliding window reset {}.", orgRecord[0], orgRecord[1], checkEntity(vecTok, orgRecord));
                vecTok.clear(); // clear existing records
                vecTok.add(orgRecord); // add the current one as the first one in the sliding window
                continue;
            }

            vecTok.add(orgRecord);

            if (vecTok.size() == window) {
                logger.trace("Entity {} time {} causes sliding window merge.", orgRecord[0], orgRecord[1]);
                String[] temporalRecord = new String[numAttributes];
                for (int i = 0; i < window; i++) {
                    orgRecord = vecTok.get(i);
                    int org = orgNumAtt - (classIncluded ? 0 : 1);
                    for (int j = 0; j < org; j++) {
                        int k = org * i + j;
                        temporalRecord[k] = new String(orgRecord[j + 2]);
                    }
                }
                if (!classIncluded) temporalRecord[numAttributes - 1] = new String(orgRecord[orgNumAtt + 1]);

                StringBuilder sb = new StringBuilder();
                for (String r: temporalRecord)
                    sb.append(r + "\t");
                logger.trace(sb.toString());

                Data.add(temporalRecord);
                vecTok.remove(0);
            }
        }
        bin.close();
        return 1;
    }

    /** Check whether the orgRecord is the same entity as the last one in vecTok
     *
     * @param vecTok the list of entities in sliding window
     * @param currRecord the current entity record
     * @return true if the current entity is the same as the sliding window
     */
    private boolean checkEntity(ArrayList<String[]> vecTok, String[] currRecord) {
        if (vecTok.isEmpty())
            return true;

        String []lastRecord = vecTok.get(vecTok.size() - 1);
        return lastRecord[0].equals(currRecord[0]);
    }

    /** Check whether the orgRecord is live for given TTL (time to live)
     *
     * @param vecTok the list of entities in sliding window
     * @param currRecord the current entity record
     * @return true if the current entity is live
     */
    private boolean checkTTL(ArrayList<String[]> vecTok, String[] currRecord) {
        if (vecTok.isEmpty())
            return true;

        String []lastRecord = vecTok.get(vecTok.size() - 1);
        return (Integer.valueOf(currRecord[1]) - Integer.valueOf(lastRecord[1])) <= ttl;
    }

}
