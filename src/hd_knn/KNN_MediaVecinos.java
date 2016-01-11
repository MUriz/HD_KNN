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
public class KNN_MediaVecinos implements KNN_method {

    @Override
    public HD_KNN.DistanceClassOutput getReducerOutput(HD_KNN.DistanceClassOutput[] nearest) {
    
        //MEDIA
        HashMap<Text, ArrayList<Double>> nearest_map = new HashMap<>();
        for (HD_KNN.DistanceClassOutput dco : nearest) {
            if (nearest_map.get(dco.instanceClass) == null) {
                nearest_map.put(dco.instanceClass, new ArrayList<Double>());
            }
            nearest_map.get(dco.instanceClass).add(dco.distance.get());
        }
        Text res_class = new Text("-1");
        double min_d = Double.MAX_VALUE;
        for (Text ic : nearest_map.keySet()) {
            double s = 0;
            for (double d : nearest_map.get(ic)) {
                s += d;
            }
            s /= nearest_map.get(ic).size();
            if (s < min_d) {
                res_class = ic;
                min_d = s;
            }
        }
        return new HD_KNN.DistanceClassOutput(res_class, new DoubleWritable(min_d));
        
    }
    
}
