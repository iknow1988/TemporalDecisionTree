package Project;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.ForkJoinPool;

public class TreeNode {
    private int level;
    private TreeNode parent;
    public TreeNode[] children = null;
    public int[] recordList;
    private double entropy;
    private LinkedList<TreeNodeGain> gainList = new LinkedList<>();
    public Map<String, Integer> classCount = null;
    public Attribute decompositionAttribute = null;
    public String[] decompositionValue = new String[4];

    public static boolean binaryDiscrete = true;
    public static Integer cores = null;
    public static ArrayList<String[]> Data; //It only consumes memory once. as it is static.
    public static Attribute[] attributes; // field, they realy exist once in memory...
    public static double topGainRatio = 0.2; // used to generated top gain list from gain list
    public static final String DELIM = "    ";

    private static final Logger LOGGER = LogManager.getLogger(TreeNode.class.getSimpleName());

    TreeNode(final ArrayList Data, final Attribute[] attributes) {
        TreeNode.attributes = attributes;
        TreeNode.Data = Data;

        this.level = 0;
        this.parent = null;
        int dataSize = Data.size();
        this.recordList = new int[dataSize];
        for (int i = 0; i < dataSize; i++)
            recordList[i] = i;
        this.entropy = calculateEntropy(this.recordList);

        LOGGER.info("Building root tree with parameters: topGainRatio = {}", topGainRatio);
        ForkJoinPool forkJoinPool = cores == null ? new ForkJoinPool() : new ForkJoinPool(cores);
        LOGGER.info("Invoking the train recursive action with parallelism = {}", forkJoinPool.getParallelism());
        forkJoinPool.invoke(new TrainRecursiveAction(this));
    }

    TreeNode(final TreeNode parent, final String[] decompositionValue, final int[] recordList) {
        this.parent = parent;
        this.level = parent.level + 1;
        this.decompositionValue = decompositionValue;
        this.recordList = recordList;
        this.entropy = calculateEntropy(this.recordList);
    }

    /**
     *
     * @return
     */
    public Collection<TreeNodeGain> choosAttributeAndDescritize() {
        for (int i = 0; i < attributes.length - 1; i++) {
            Attribute att = attributes[i];
            // TODO: why we could not re-use an attribute in the tree??
            if (!alreadyUsed(att) && checkTime(att)) {
                int[][] tempSubsets = descritize(recordList, att);
                double tempGain = calculateGain(tempSubsets);
                gainList.add(new TreeNodeGain(att, tempGain, tempSubsets));
            }
        }

        if (gainList.size() == 0) {
            return Collections.singletonList(new TreeNodeGain(null, 0, null));
        }

        Collections.sort(gainList, new Comparator<TreeNodeGain>() {
            @Override
            public int compare(final TreeNodeGain l, final TreeNodeGain r) {
             if (l.gain > r.gain) return -1;
             else if (l.gain == r.gain) return 0;
             else return 1;
            }
        });
        //Add by L and Qiezi. In addition of simply choosing the feature with highest gain,
        //We choose the feature with other time tag and nearest to the max-gain feature.
        //topGainRatio is customised to choose the top say, 20% of the original feature list. You can change its value if you want to.
        List<TreeNodeGain> topGainList = gainList.subList(0, (int) Math.ceil(topGainRatio * gainList.size()));
        Collections.sort(topGainList, new Comparator<TreeNodeGain>() {
            @Override
            public int compare(final TreeNodeGain l, final TreeNodeGain r) {
                if (l.att == r.att) return 0;
                else if (l.att == null) return -1;
                else if (r.att == null) return 1;
                else if (l.att.time > r.att.time) return 1;
                else if (l.att.time == r.att.time) {
                    if (l.gain > r.gain) return -1;
                    else if (l.gain == r.gain) return 0;
                    else return 1;
                } else return -1;
            }
        });

        return topGainList;
    }

    public boolean alreadyUsed(Attribute att) {
        if (children != null) {
            if (decompositionAttribute.equalOrder(att))
                return true;
        }
        if (parent == null) return false;

        return parent.alreadyUsed(att);
    }


    private boolean checkTime(Attribute att) {
        if (parent == null) return true;
        if (parent.decompositionAttribute.later(att)) return false;
        return true;
    }


    private int[][] descritize(int[] recordList, Attribute att) {

        if (recordList.length == 0 || recordList == null) {
            LOGGER.fatal("recordList empty or null");
            System.exit(1);
        }

        TreeMap<String, ArrayList<Integer>> set = new TreeMap();

        for (int record : recordList) {
            String value = Data.get(record)[att.order];
            if (set.containsKey(value))
                set.get(value).add(record);
            else {
                ArrayList<Integer> l;
                set.put(value, l = new ArrayList());
                l.add(record);
            }
        }

        String[] attValues = set.keySet().toArray(new String[0]);

        if (att.nominal || attValues.length <= 2) {

            int[][] subsets = new int[attValues.length][];
            Integer[] list;
            for (int i = 0; i < attValues.length; i++) {
                list = set.get(attValues[i]).toArray(new Integer[0]);
                subsets[i] = new int[list.length];
                for (int j = 0; j < list.length; j++) {
                    subsets[i][j] = list[j];
                }
            }
            return subsets;
        }

        int[] bestTailList = null;
        int[] bestHeadList = null;
        double minTailEntropy = 0;
        double minHeadEntropy = 0;
        double minTarEntropy = Double.MAX_VALUE;
        for (int i = 1; i < attValues.length - 1; i++) {
            String value = attValues[i];
            ArrayList<Integer>[] tail = set.tailMap(value).values().toArray(new ArrayList[0]);
            ArrayList<Integer>[] head = set.headMap(value).values().toArray(new ArrayList[0]);
            ArrayList<Integer> tailRecur = new ArrayList();
            ArrayList<Integer> headRecur = new ArrayList();
            for (ArrayList<Integer> l : tail) for (int in : l) tailRecur.add(in);
            for (ArrayList<Integer> l : head) for (int in : l) headRecur.add(in);
            int[] tailList = new int[tailRecur.size()];
            int[] headList = new int[headRecur.size()];
            for (int j = 0; j < tailRecur.size(); j++) tailList[j] = tailRecur.get(j);
            for (int j = 0; j < headRecur.size(); j++) headList[j] = headRecur.get(j);
            Double tailEntropy = calculateEntropy(tailList);
            Double headEntropy = calculateEntropy(headList);
            Double tarEntropy = (double) (tailEntropy * (double) tailList.length + headEntropy * (double) headList.length) / (double) recordList.length;
            if (tarEntropy < minTarEntropy) {
                minTarEntropy = tarEntropy;
                bestTailList = tailList;
                bestHeadList = headList;
                minTailEntropy = tailEntropy;
                minHeadEntropy = headEntropy;
            }
        }
        int[][] subsets;

        if (binaryDiscrete) {
            subsets = new int[2][];
            subsets[0] = bestHeadList;
            subsets[1] = bestTailList;
            return subsets;
        }

        int N = recordList.length;
        int K = classValues(recordList).size();
        int K1 = classValues(bestTailList).size();
        int K2 = classValues(bestHeadList).size();
        double listEntropy = calculateEntropy(recordList);
        double gainLimit = ((Math.log(N - 1) + Math.log(Math.pow(3, K) - 2)) / Math.log(2)
                - ((double) K) * listEntropy + ((double) K1) * minTailEntropy + ((double) K2) * minHeadEntropy) / (double) N;

        if (listEntropy - minTarEntropy > gainLimit) {
            int[][] subsets1 = descritize(bestHeadList, att);
            int[][] subsets2 = descritize(bestTailList, att);
            int size1 = subsets1.length;
            int size2 = subsets2.length;
            subsets = new int[size1 + size2][];
            int i = 0;
            for (int[] l : subsets1) {
                subsets[i] = l;
                i++;
            }
            for (int[] l : subsets2) {
                subsets[i] = l;
                i++;
            }
        } else {
            subsets = new int[1][];
            subsets[0] = recordList;
        }
        return subsets;

    }


    private double calculateGain(int[][] subsets) {

        if (subsets.length == 1) return 0;
        double gain = entropy;
        for (int[] l : subsets) {
            gain -= calculateEntropy(l) * ((double) l.length / (double) recordList.length);
        }
        return gain;
    }


    private HashSet classValues(int[] recordList) {
        HashSet set = new HashSet();
        for (int i : recordList) set.add(Data.get(i)[attributes.length - 1]);
        return set;
    }


    public String printTree(String tab) throws Exception {

        if (this.parent != null) {
            throw new Exception("This is not root node");
        }
        return printTree(this, tab);
    }

    public String printTree(TreeNode node, String tab) {

        String treeStr = "";
        int classAtt = attributes.length - 1;
        HashSet<String> set = classValues(node.recordList);

        if (node.children == null) {
            if (set.size() == 1) {
                treeStr += (tab + TreeNode.DELIM + attributes[classAtt] + " = \""
                        + Data.get(node.recordList[0])[classAtt] + "\";");
                treeStr += "\n";
                return treeStr;
            }
            treeStr += (tab + TreeNode.DELIM + attributes[classAtt] + " = {");
            for (String classValue : set) {
                treeStr += ("\"" + classValue + "\",");
            }
            treeStr += ("};");
            treeStr += "\n";
            return treeStr;
        }

        int numvalues = node.children.length;
        for (int i = 0; i < numvalues; i++) {
            if (node.decompositionAttribute.nominal) {
                String symbol = node.children[i].decompositionValue[1];
                treeStr += (tab + "if( " + attributes[node.decompositionAttribute.order] + " =\"" + symbol + "\") {");
            } else {
                String leftThreshold = node.children[i].interval(0);
                String leftThreshold1 = leftThreshold.equals(Double.toString(-1 * Double.MAX_VALUE)) ? "-infinite" : leftThreshold;
                String rightThreshold = node.children[i].interval(1);
                String rightThreshold1 = rightThreshold.equals(Double.toString(Double.MAX_VALUE)) ? "+infinite" : rightThreshold;
                if (!treeStr.endsWith(" else "))
                    treeStr += tab;
                treeStr += ("if( " + leftThreshold1 + " <= " + attributes[node.decompositionAttribute.order] + " < " + rightThreshold1 + ") {");
            }
            //print the metric value in each node.
            treeStr += "\t[" + new DecimalFormat("#.00").format(node.getAccuracy()) + "]\n";
            treeStr += printTree(node.children[i], tab + TreeNode.DELIM);
            if (i != numvalues - 1)
                treeStr += (tab + "} else ");
            else {
                treeStr += (tab + "}");
                treeStr += "\n";
            }
        }
        return treeStr;
    }

    public TreeNode predict(String[] record) {
        LOGGER.trace("Predicting record: {}", record);

        if (children == null) return this;

        for (TreeNode child : children) {
            if (decompositionAttribute.nominal) {
                if (record[decompositionAttribute.order].equals(child.decompositionValue[1]))

                    return child.predict(record);
            } else {
                if (Double.parseDouble(child.interval(0)) <= Double.parseDouble(record[decompositionAttribute.order])
                        && Double.parseDouble(record[decompositionAttribute.order]) < Double.parseDouble(child.interval(1)))

                    return child.predict(record);
            }
        }
        return null;
    }

    public double predictSelfAccuracy() {
        ArrayList<String[]> recordData = new ArrayList<>();
        for (int i : recordList)
            recordData.add(Data.get(i));
        DecisionTree tmpDecisionTree = new DecisionTree();
        tmpDecisionTree.Data = recordData;
        tmpDecisionTree.attributes = attributes;
        return tmpDecisionTree.predictDataset(this);
    }

    public double getEntropy() {
        return this.entropy;
    }

    public int getRecordSize() {
        return this.recordList.length;
    }

    public int getRecord(int index) {
        if (index < 0 || index >= getRecordSize()) {
            LOGGER.error("Invalid index {} getting record from tree node.", index);
            return -1;
        } else {
            return this.recordList[index];
        }
    }

    private double calculateEntropy(int[] recordList) {

        int numdata = recordList.length;
        if (numdata == 0 || numdata == 1) return 0;
        int classAttribute = attributes.length - 1;

        TreeMap<String, Integer> set = new TreeMap();

        for (int i : recordList) {
            String classValue = Data.get(i)[classAttribute];
            set.put(classValue, set.keySet().contains(classValue) ? set.get(classValue) + 1 : 1);
        }

        double sum = 0;

        for (String value : set.keySet()) {
            int count = set.get(value);
            double probability = (double) count / (double) numdata;
            if (count > 0) sum = sum - probability * (Math.log(probability) / Math.log(2));
        }

        return sum;
    }

    /**
     * Calculate the metric value in each node, so that we can evaluate the performance of a picked feature.
     * @author L
     */
    public double getAccuracy() {
        Map<String, Integer> counts = new HashMap<>();
        for (int i : recordList) {
            String attr = Data.get(i)[attributes.length - 1];
            counts.put(attr, counts.containsKey(attr) ? counts.get(attr) + 1 : 1);
        }
        int max_count = 0;
        for (int c : counts.values()) {
            if (c > max_count)
                max_count = c;
        }
        return max_count * 1.0 / recordList.length;
    }

    private String interval(int ch) {
        String[] value = this.decompositionValue;
        double start = value[0] == null ? -1 * Double.MAX_VALUE : (Double.parseDouble(value[0]) + Double.parseDouble(value[1])) / 2;
        double end = value[3] == null ? Double.MAX_VALUE : (Double.parseDouble(value[2]) + Double.parseDouble(value[3])) / 2;
        return Double.toString(ch == 0 ? start : end);
    }

    /** Print the distribution of the target value in one tree node.
     */
    private String getTargetStr() {
        int t = 0;
        for (int i : recordList)
            if (Data.get(i)[attributes.length - 1].equals("1"))
                t++;
        return " [" + t + "/" + (recordList.length - t) + "] ";
    }

    /** monitorGain is just used to present the gain value of features in a descending order.
     */
    private String getGainStr() {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        for (TreeNodeGain tng : this.gainList)
            sb.append(tng.att + ":" + tng.gain + " ");
        sb.append("}");
        return sb.toString();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        sb.append("l: ");
        sb.append(level);
        sb.append(", e: ");
        sb.append(new DecimalFormat("#.00").format(entropy));
        if (decompositionAttribute != null) {
            sb.append(", att: ");
            sb.append(decompositionAttribute.name);
            sb.append(", accuracy: ");
            sb.append(getAccuracy());
        }
        sb.append("]");
        return sb.toString();
    }
}
