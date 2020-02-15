
package Project;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import javax.swing.ButtonGroup;
import javax.swing.JFileChooser;
import javax.swing.UIManager;


public class GUI extends javax.swing.JFrame {

    File trainFile = null;
    int win = 1;
    File testFile = null;
    TreeNode root = null;
    JFileChooser fc = new JFileChooser();
    /**
     * Creates new form IF
     */
    ButtonGroup group = new ButtonGroup();

    private static final Logger LOGGER = LogManager.getLogger(GUI.class.getSimpleName());

    public GUI() {
        initComponents();
        group.add(this.jRadioButton1);
        group.add(this.jRadioButton2);
        group.add(this.jRadioButton3);
    }

    /**
     * This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jTextField2 = new javax.swing.JTextField();
        jPanel1 = new javax.swing.JPanel();
        jPanel3 = new javax.swing.JPanel();
        jButton1 = new javax.swing.JButton();
        jTextField1 = new javax.swing.JTextField();
        jButton3 = new javax.swing.JButton();
        jPanel9 = new javax.swing.JPanel();
        jLabel2 = new javax.swing.JLabel();
        jPanel5 = new javax.swing.JPanel();
        jPanel8 = new javax.swing.JPanel();
        jPanel7 = new javax.swing.JPanel();
        jTextField5 = new javax.swing.JTextField();
        jPanel13 = new javax.swing.JPanel();
        jLabel1 = new javax.swing.JLabel();
        jTextField4 = new javax.swing.JTextField();
        jCheckBox5 = new javax.swing.JCheckBox();
        jCheckBox4 = new javax.swing.JCheckBox();
        jPanel4 = new javax.swing.JPanel();
        jButton2 = new javax.swing.JButton();
        jTextField3 = new javax.swing.JTextField();
        jButton5 = new javax.swing.JButton();
        jPanel6 = new javax.swing.JPanel();
        jCheckBox9 = new javax.swing.JCheckBox();
        jPanel12 = new javax.swing.JPanel();
        jRadioButton1 = new javax.swing.JRadioButton();
        jRadioButton2 = new javax.swing.JRadioButton();
        jRadioButton3 = new javax.swing.JRadioButton();
        jCheckBox6 = new javax.swing.JCheckBox();
        jPanel2 = new javax.swing.JPanel();
        jPanel10 = new javax.swing.JPanel();
        jScrollPane1 = new javax.swing.JScrollPane();
        jTextArea1 = new javax.swing.JTextArea();
        jPanel11 = new javax.swing.JPanel();
        jButton4 = new javax.swing.JButton();

        jTextField2.setText("jTextField2");

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Temporal Classifier");
        setMinimumSize(new java.awt.Dimension(800, 600));

        jPanel1.setLayout(new javax.swing.BoxLayout(jPanel1, javax.swing.BoxLayout.Y_AXIS));

        jPanel3.setBorder(javax.swing.BorderFactory.createEmptyBorder(10, 10, 1, 10));
        jPanel3.setLayout(new javax.swing.BoxLayout(jPanel3, javax.swing.BoxLayout.LINE_AXIS));

        jButton1.setText("Browse");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });
        jPanel3.add(jButton1);

        jTextField1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jTextField1ActionPerformed(evt);
            }
        });
        jPanel3.add(jTextField1);

        jButton3.setText("Train");
        jButton3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton3ActionPerformed(evt);
            }
        });
        jPanel3.add(jButton3);

        jPanel1.add(jPanel3);

        jPanel9.setBorder(javax.swing.BorderFactory.createEmptyBorder(10, 10, 1, 10));
        jPanel9.setLayout(new javax.swing.BoxLayout(jPanel9, javax.swing.BoxLayout.LINE_AXIS));

        jLabel2.setText("Nominals:");
        jPanel9.add(jLabel2);

        jPanel5.setBorder(javax.swing.BorderFactory.createEmptyBorder(1, 1, 1, 11));
        jPanel5.setLayout(new javax.swing.BoxLayout(jPanel5, javax.swing.BoxLayout.LINE_AXIS));

        jPanel8.setLayout(new javax.swing.BoxLayout(jPanel8, javax.swing.BoxLayout.LINE_AXIS));
        jPanel5.add(jPanel8);

        jPanel7.setLayout(new javax.swing.BoxLayout(jPanel7, javax.swing.BoxLayout.LINE_AXIS));
        jPanel7.add(jTextField5);
        jPanel7.add(jPanel13);

        jLabel1.setText("Window Size:");
        jPanel7.add(jLabel1);

        jTextField4.setText("1");
        jPanel7.add(jTextField4);

        jPanel5.add(jPanel7);

        jCheckBox5.setText("Include Class");
        jCheckBox5.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jCheckBox5ActionPerformed(evt);
            }
        });
        jPanel5.add(jCheckBox5);

        jCheckBox4.setSelected(true);
        jCheckBox4.setText("Binary Discrete");
        jPanel5.add(jCheckBox4);

        jPanel9.add(jPanel5);

        jPanel1.add(jPanel9);

        jPanel4.setBorder(javax.swing.BorderFactory.createEmptyBorder(10, 10, 1, 10));
        jPanel4.setLayout(new javax.swing.BoxLayout(jPanel4, javax.swing.BoxLayout.LINE_AXIS));

        jButton2.setText("Browse");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });
        jPanel4.add(jButton2);

        jTextField3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jTextField3ActionPerformed(evt);
            }
        });
        jPanel4.add(jTextField3);

        jButton5.setText("Test");
        jButton5.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton5ActionPerformed(evt);
            }
        });
        jPanel4.add(jButton5);

        jPanel1.add(jPanel4);

        jCheckBox9.setText("Bayesian");
        jCheckBox9.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jCheckBox9ActionPerformed(evt);
            }
        });
        jCheckBox9.addPropertyChangeListener(new java.beans.PropertyChangeListener() {
            public void propertyChange(java.beans.PropertyChangeEvent evt) {
                jCheckBox9PropertyChange(evt);
            }
        });
        jPanel6.add(jCheckBox9);

        jPanel12.setEnabled(false);
        jPanel12.setLayout(new javax.swing.BoxLayout(jPanel12, javax.swing.BoxLayout.LINE_AXIS));

        jRadioButton1.setSelected(true);
        jRadioButton1.setText("Randomly");
        jRadioButton1.setEnabled(false);
        jPanel12.add(jRadioButton1);

        jRadioButton2.setText("Strict");
        jRadioButton2.setEnabled(false);
        jPanel12.add(jRadioButton2);

        jRadioButton3.setText("Loose");
        jRadioButton3.setEnabled(false);
        jPanel12.add(jRadioButton3);

        jPanel6.add(jPanel12);

        jCheckBox6.setText("Update");
        jPanel6.add(jCheckBox6);

        jPanel1.add(jPanel6);

        getContentPane().add(jPanel1, java.awt.BorderLayout.PAGE_START);

        jPanel2.setBorder(javax.swing.BorderFactory.createEmptyBorder(10, 10, 10, 10));
        jPanel2.setLayout(new javax.swing.BoxLayout(jPanel2, javax.swing.BoxLayout.LINE_AXIS));

        jPanel10.setLayout(new java.awt.BorderLayout());

        jScrollPane1.setBorder(new javax.swing.border.LineBorder(new java.awt.Color(0, 0, 0), 1, true));

        jTextArea1.setColumns(20);
        jTextArea1.setRows(5);
        jScrollPane1.setViewportView(jTextArea1);

        jPanel10.add(jScrollPane1, java.awt.BorderLayout.CENTER);

        jPanel11.setLayout(new java.awt.FlowLayout(java.awt.FlowLayout.RIGHT));

        jButton4.setText("Clear");
        jButton4.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton4ActionPerformed(evt);
            }
        });
        jPanel11.add(jButton4);

        jPanel10.add(jPanel11, java.awt.BorderLayout.PAGE_END);

        jPanel2.add(jPanel10);

        getContentPane().add(jPanel2, java.awt.BorderLayout.CENTER);

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jTextField1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jTextField1ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_jTextField1ActionPerformed

    private void jButton4ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton4ActionPerformed
        // TODO add your handling code here:
        jTextArea1.setText("");
    }//GEN-LAST:event_jButton4ActionPerformed

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        // TODO add your handling code here:
        fc.setDialogTitle("Train File");
        int i = fc.showDialog(this, "Select");
        if (i == JFileChooser.APPROVE_OPTION) {
            trainFile = fc.getSelectedFile();
            testFile = trainFile;
            this.jTextField1.setText(trainFile.getPath());
            this.jTextField3.setText(trainFile.getPath());
        }
    }//GEN-LAST:event_jButton1ActionPerformed

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        // TODO add your handling code here:
        fc.setDialogTitle("Test File");
        if (fc.showDialog(this, "Select") == JFileChooser.APPROVE_OPTION) {
            testFile = fc.getSelectedFile();
            jTextField3.setText(testFile.getPath());
        }
    }//GEN-LAST:event_jButton2ActionPerformed

    private void jButton3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton3ActionPerformed
        // TODO add your handling code here:

        if (trainFile == null) {
            jTextArea1.append("Please choose the train file.\n");
            return;
        }
        DecisionTree me1 = null;
        me1 = new DecisionTree();
        me1.fileName = trainFile.getPath();
        String nomi = jTextField5.getText();
        if (!nomi.equals("")) {
            String[] nominals = nomi.split("(\\s*[,;./]\\s*)");
            for (String n : nominals) {
                System.out.println(n + " is nominal\n");
                me1.nominals.add(Integer.parseInt(n));
            }
        }

        win = me1.window = Integer.parseInt(jTextField4.getText());
        // me1.bayesian=this.jCheckBox3.isSelected();
        me1.classIncluded = this.jCheckBox5.isSelected();
        // me1.randomly=this.jCheckBox2.isSelected();
        // me1.strict=this.jCheckBox1.isSelected();
        // me1.update=this.jCheckBox3.isSelected();
        TreeNode.binaryDiscrete = this.jCheckBox4.isSelected();
        long startTime = System.currentTimeMillis();
        int status = -1;

        try {
            status = me1.readData_new();
        } catch (Exception ex) {
            LOGGER.fatal("Could not read training data");
            ex.printStackTrace();
        }
        if (status <= 0) return;


        root = new TreeNode(me1.Data, me1.attributes);
        long endTime = System.currentTimeMillis();
        long totalTime = (endTime - startTime) / 1000;
        jTextArea1.append(totalTime + " Seconds\n\n");
        try {
            jTextArea1.append(root.printTree("") + "\n");
        } catch (Exception ex) {
            LOGGER.error("Could not print tree to the tex area 1");
            ex.printStackTrace();
        }
        // jTextArea1.append("\n"+me1.predictDataset(root));

    }//GEN-LAST:event_jButton3ActionPerformed

    private void jButton5ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton5ActionPerformed
        // TODO add your handling code here:
        if (testFile == null) {
            jTextArea1.append("Test File has not been chosen yet!\n");
        } else if (root != null) {
            DecisionTree me1 = new DecisionTree();
            me1.window = win;
            DecisionTree.bayesian = this.jCheckBox9.isSelected();
            me1.classIncluded = this.jCheckBox5.isSelected();
            me1.update = this.jCheckBox6.isSelected();
            me1.fileName = this.jTextField3.getText();

            if (jRadioButton1.isSelected()) {
                me1.randomly = true;
                me1.strict = false;
            } else if (jRadioButton2.isSelected()) {
                me1.randomly = false;
                me1.strict = true;
            } else if (jRadioButton3.isSelected()) {
                me1.randomly = false;
                me1.strict = false;
            }
            TreeNode.binaryDiscrete = this.jCheckBox4.isSelected();
            try {


                me1.readData_new();
            } catch (Exception ex) {
                LOGGER.fatal("jButton5ActionPerformed - Could not read data me 1");
            }
            double average = 0;
            //int n = 100;
            int n = 10;
            for (int i = 0; i < n; i++) {
                double pred = me1.predictDataset(root);
                jTextArea1.append("%" + round(pred) + "\n");
                average += pred / ((double) n);
            }
            jTextArea1.append("\naverage: %" + round(average) + "\n--------\n");
        } else jTextArea1.append("The tree has not been trained yet!\n");

    }//GEN-LAST:event_jButton5ActionPerformed

    private void jTextField3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jTextField3ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_jTextField3ActionPerformed

    private void jCheckBox5ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jCheckBox5ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_jCheckBox5ActionPerformed

    private void jCheckBox9PropertyChange(java.beans.PropertyChangeEvent evt) {//GEN-FIRST:event_jCheckBox9PropertyChange
        // TODO add your handling code here:
    }//GEN-LAST:event_jCheckBox9PropertyChange

    private void jCheckBox9ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jCheckBox9ActionPerformed
        // TODO add your handling code here:
        if (jCheckBox9.isSelected()) {
            this.jRadioButton1.setEnabled(true);
            this.jRadioButton2.setEnabled(true);
            this.jRadioButton3.setEnabled(true);
        } else {
            this.jRadioButton1.setEnabled(false);
            this.jRadioButton2.setEnabled(false);
            this.jRadioButton3.setEnabled(false);
        }
    }//GEN-LAST:event_jCheckBox9ActionPerformed

    public static void launch() {
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                try {
                    UIManager.setLookAndFeel("com.sun.java.swing.plaf.windows.WindowsLookAndFeel");

                } catch (Exception ex) {
                    LOGGER.warn("UIManager could not set look and feel");
                }
                new GUI().setVisible(true);
                LOGGER.info("GUI launched successfully.");
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    private javax.swing.JButton jButton4;
    private javax.swing.JButton jButton5;
    private javax.swing.JCheckBox jCheckBox4;
    private javax.swing.JCheckBox jCheckBox5;
    private javax.swing.JCheckBox jCheckBox6;
    private javax.swing.JCheckBox jCheckBox9;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel10;
    private javax.swing.JPanel jPanel11;
    private javax.swing.JPanel jPanel12;
    private javax.swing.JPanel jPanel13;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JPanel jPanel3;
    private javax.swing.JPanel jPanel4;
    private javax.swing.JPanel jPanel5;
    private javax.swing.JPanel jPanel6;
    private javax.swing.JPanel jPanel7;
    private javax.swing.JPanel jPanel8;
    private javax.swing.JPanel jPanel9;
    private javax.swing.JRadioButton jRadioButton1;
    private javax.swing.JRadioButton jRadioButton2;
    private javax.swing.JRadioButton jRadioButton3;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JTextArea jTextArea1;
    private javax.swing.JTextField jTextField1;
    private javax.swing.JTextField jTextField2;
    private javax.swing.JTextField jTextField3;
    private javax.swing.JTextField jTextField4;
    private javax.swing.JTextField jTextField5;
    // End of variables declaration//GEN-END:variables

    private String round(double d) {
        return "" + (int) d + "." + (int) ((d - (double) (int) d) * 100);
    }
}
