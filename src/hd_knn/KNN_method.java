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
public interface KNN_method {
    
    public HD_KNN.DistanceClassOutput getReducerOutput(HD_KNN.DistanceClassOutput[] nearest);
    
}
