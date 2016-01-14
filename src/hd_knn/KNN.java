/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hd_knn;

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

/**
 *
 * @author mikel
 */
public class KNN {
    
    private static ArrayList<String> readFile(String filename) throws FileNotFoundException, IOException {
        
        ArrayList<String> lines = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String sCurrentLine;
        while ((sCurrentLine = br.readLine()) != null) {
            lines.add(sCurrentLine);
        }
        return lines;
        
    }
    
    public static void main(String[] args) throws IOException {
        
        ArrayList<String> train = readFile("knn_train.txt");
        ArrayList<String> test = readFile("knn_test.txt");
        for (int i = 0; i < test.size(); i++) {
            String ins_class = getClass(test.get(i), train);
            System.out.println(test.get(i) + "\t" + ins_class);
        }
        
    }
    
    public static String getClass(String test, ArrayList<String> train) {
        
        String[] test_car = test.split(",");
        String inst_class = "-1";
        double min_dist  = Double.MAX_VALUE;
        for (int i = 0; i < train.size(); i++) {
            String[] train_car = train.get(i).split(",");
            double distance = getDistance(test_car, train_car);
            if (distance < min_dist) {
                min_dist = distance;
                inst_class = train_car[train_car.length -1];
            }
        }
        return inst_class + "\t" + String.valueOf(min_dist);
        
    }
    
    public static double getDistance(String[] test, String[] train) {
        
        double d = 0;
        for (int i = 0; i < test.length - 1; i++) {
            double v1 = Double.parseDouble(test[i]);
            double v2 = Double.parseDouble(train[i]);
            d += (v1- v2)*(v1 - v2);
        }
        return Math.sqrt(d);
        
    }
    
}
