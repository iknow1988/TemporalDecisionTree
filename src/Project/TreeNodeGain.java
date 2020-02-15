package Project;

import java.text.DecimalFormat;

public class TreeNodeGain {
    public int[][] subsets;
    public double gain;
    public Attribute att;

    public TreeNodeGain(Attribute att, double gain, int[][] subsets) {
        this.att = att;
        this.gain = gain;
        this.subsets = subsets;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        sb.append("gain: ");
        sb.append(new DecimalFormat("#.00").format(gain));
        if (att != null) {
            sb.append(", att: ");
            sb.append(att);
        }
        sb.append("]");
        return sb.toString();
    }
}