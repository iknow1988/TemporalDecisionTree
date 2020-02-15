package Project;

public class Attribute {

    String name = null;
    int time = 0;
    int order = 0;
    boolean nominal = false;

    Attribute(String name, int time, int order, boolean nominal) {
        this.name = new String(name);
        this.time = time;
        this.order = order;
        this.nominal = nominal;
    }

    boolean equalOrder(Attribute att) {
        if (att.order == this.order) return true;
        return false;
    }

    boolean later(Attribute att) {
        if (this.time > att.time) return true;
        return false;
    }

    public String toString() {
        return name + "_" + time;
    }
}
