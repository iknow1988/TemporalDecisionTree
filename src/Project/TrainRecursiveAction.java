package Project;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.RecursiveTask;

/**
 * This is the recursive and parallel task to create the tree node
 * @author L
 */
public class TrainRecursiveAction extends RecursiveAction {
    private TreeNode treeNode;

    private static final Logger LOGGER = LogManager.getLogger(TrainRecursiveAction.class.getSimpleName());

    public TrainRecursiveAction(TreeNode treeNode) {
        this.treeNode = treeNode;
    }

    @Override
    protected void compute() {
        if (treeNode.getEntropy() == 0) {
            // This is a perfect leaf node
            LOGGER.trace("Found a perfect leaf node {}", treeNode);
            return;
        }
        double maxAccuracy = 0;
        Attribute maxDecompositionAttribute = null;
        Map<String, Integer> maxClassCount = null;
        TreeNode[] maxChildren = null;

        Collection<TreeNodeGain> treeNodeGainCollection = treeNode.choosAttributeAndDescritize();
        for (TreeNodeGain tng : treeNodeGainCollection) {
            LOGGER.trace("In {} chances, split the node {} further with TreeNodeGain: {}", treeNodeGainCollection.size(), this, tng);
            double currentAccuracy;
            treeNode.decompositionAttribute = tng.att;

            int[][] subsets = tng.subsets;
            if (subsets == null) {
                // This is a leaf node, save the context for decision maker
                treeNode.classCount = new TreeMap<>();
                for (int rec : treeNode.recordList) {
                    String classVal = TreeNode.Data.get(rec)[TreeNode.attributes.length - 1];
                    treeNode.classCount.put(classVal, treeNode.classCount.containsKey(classVal) ? treeNode.classCount.get(classVal) + 1 : 1);
                }
                currentAccuracy = treeNode.getAccuracy();
            } else {
                // This is an inner node, let us construct its children
                int numIntervals = subsets.length;
                String[][] intervals = new String[numIntervals][];

                treeNode.children = new TreeNode[numIntervals];

                int i = 0;
                List<RecursiveAction> recursiveTasks = new LinkedList<>();
                for (int[] records : subsets) {

                    intervals[i] = new String[4];

                    intervals[i][0] = i == 0 ? null : intervals[i - 1][2];
                    intervals[i][1] = TreeNode.Data.get(records[0])[treeNode.decompositionAttribute.order];
                    intervals[i][2] = TreeNode.Data.get(records[records.length - 1])[treeNode.decompositionAttribute.order];
                    intervals[i][3] = (i == numIntervals - 1) ? null : TreeNode.Data.get(subsets[i + 1][0])[treeNode.decompositionAttribute.order];

                    treeNode.children[i] = new TreeNode(treeNode, intervals[i], subsets[i]);
                    TrainRecursiveAction childRecursiveTask = new TrainRecursiveAction(treeNode.children[i]);
                    recursiveTasks.add(childRecursiveTask);
                    childRecursiveTask.fork();

                    i++;
                }

                for (RecursiveAction task : recursiveTasks) {
                    task.join();
                }

                currentAccuracy = treeNode.predictSelfAccuracy();
            }

            /* Decide whether use current tree or not. Need to save all the context */
            if (currentAccuracy >= maxAccuracy) {
                maxDecompositionAttribute = treeNode.decompositionAttribute;
                maxClassCount = treeNode.classCount;
                maxChildren = treeNode.children;
            }
        }

        // Of course we should use the max metric
        treeNode.decompositionAttribute = maxDecompositionAttribute;
        treeNode.classCount = maxClassCount;
        treeNode.children = maxChildren;

        LOGGER.debug("Constructed the node {} from {} candidates", treeNode, treeNodeGainCollection.size());
    }

}
