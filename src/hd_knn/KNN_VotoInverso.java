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
public class KNN_VotoInverso implements KNN_method {

    @Override
    public HD_KNN.DistanceClassOutput getReducerOutput(HD_KNN.DistanceClassOutput[] nearest) {
        
        //MEDIA
        HashMap<Text, ArrayList<Double>> nearest_map = new HashMap<>();
        for (HD_KNN.DistanceClassOutput dco : nearest) {
            if (nearest_map.get(dco.instanceClass) == null) {
                nearest_map.put(dco.instanceClass, new ArrayList<Double>());
            }
            if (dco.distance.get() == 0) {
                nearest_map.get(dco.instanceClass).add(Double.MAX_VALUE);
            } else {
                nearest_map.get(dco.instanceClass).add(1/dco.distance.get());
            }
        }
        Text res_class = new Text("-1");
        double max = -1;
        for (Text ic : nearest_map.keySet()) {
            double s = 0;
            for (double d : nearest_map.get(ic)) {
                s += d;
            }
            if (s > max) {
                res_class = ic;
                max = s;
            }
        }
        return new HD_KNN.DistanceClassOutput(res_class, new DoubleWritable(max));
        
    }
    
}
