/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hd_knn;

import java.util.ArrayList;
import java.util.HashMap;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;

/**
 *
 * @author mikel
 */
public class KNN_Normal implements KNN_method {

    @Override
    public HD_KNN.DistanceClassOutput getReducerOutput(HD_KNN.DistanceClassOutput[] nearest) {
        
        HashMap<String, Integer> nearest_map = new HashMap<>();
        HashMap<String, Double> distances = new HashMap<>();
        for (HD_KNN.DistanceClassOutput dco : nearest) {
                nearest_map.put(dco.instanceClass.toString(), 0);
                distances.put(dco.instanceClass.toString(), Double.MAX_VALUE);
        }
        for (HD_KNN.DistanceClassOutput dco : nearest) {
                double dist = distances.get(dco.instanceClass.toString());
                if (dco.distance.get() < dist) {
                    distances.put(dco.instanceClass.toString(), dco.distance.get());
                }
                nearest_map.put(dco.instanceClass.toString(), nearest_map.get(dco.instanceClass.toString()) + 1);
        }
        Text res_class = new Text("-1");
        int max = -1;
        for (String ic : nearest_map.keySet()) {
            int n = nearest_map.get(ic);
            if (n > max) {
                res_class = new Text(ic);
                max = n;
            }
        }
        return new HD_KNN.DistanceClassOutput(res_class, new DoubleWritable(distances.get(res_class.toString())));
        
        
    }
    
}
