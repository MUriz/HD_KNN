/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hd_knn;

/**
 *
 * @author mikel
 */
public class KNN_ChebyshevDistance implements KNN_distance {

    @Override
    public double distance(String[] train, String[] test) {
        
        double s = -Double.MAX_VALUE;
        for (int i = 0; i < test.length; i++) {
            try {
                double val1 = Double.valueOf(train[i]);
                double val2 = Double.valueOf(test[i]);
                double d = Math.abs(val1-val2);
                if ( d > s) {
                    s = d;
                }
            } catch (Exception e) {

            }

        }
        return s;
        
    }
    
}
