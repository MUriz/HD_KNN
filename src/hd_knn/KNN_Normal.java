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
        
        HashMap<Text, Integer> nearest_map = new HashMap<>();
        for (HD_KNN.DistanceClassOutput dco : nearest) {
                nearest_map.put(dco.instanceClass, 0);
        }
        for (HD_KNN.DistanceClassOutput dco : nearest) {
                nearest_map.put(dco.instanceClass, nearest_map.get(dco.instanceClass) + 1);
        }
        Text res_class = new Text("-1");
        int max = -1;
        for (Text ic : nearest_map.keySet()) {
            int n = nearest_map.get(ic);
            if (n > max) {
                res_class = ic;
                max = n;
            }
        }
        return new HD_KNN.DistanceClassOutput(res_class, new DoubleWritable(max));
        
        
    }
    
}
