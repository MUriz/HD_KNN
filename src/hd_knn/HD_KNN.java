/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hd_knn;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class HD_KNN {

    public static class DistanceCalculatorMapper extends Mapper<Object, Text, Text, Text>{

    private String readTest(Configuration conf) {
        
        try{
            String file_path = "/home/hadoop/";
            Path pt = new Path(file_path + "/" + "knn_test.txt");
            FileSystem fs = FileSystem.get( new URI(file_path), conf);
            LocalFileSystem localFileSystem = fs.getLocal(conf);
            BufferedReader bufferRedaer = new BufferedReader(new InputStreamReader(localFileSystem.open(pt)));

            String str = null;
            StringBuilder str_build = new StringBuilder();
            while ((str = bufferRedaer.readLine())!= null)
            {
                str_build.append(str);
            }
            return str_build.toString();
        }catch(Exception e){
            e.printStackTrace();
        }
        return null;
    }
      
    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        
        // In value train instances
        // x11,x12,...,x1m
        // x21,x22,...,x2m
        // ...............
        // xn1,xn2,...,xnm
        //Separate train data by "\n"
        StringTokenizer trainInstances = new StringTokenizer(value.toString(), "\n");
        while (trainInstances.hasMoreTokens()) {
            // Separate data by ','
            StringTokenizer dataValues = new StringTokenizer(trainInstances.nextToken(), ",");
            //Convert string to double
            double[] trainData = new double[dataValues.countTokens() -1 ];
            String trainClass;
            int i = 0;
            while (dataValues.hasMoreTokens()) {
                // If is the last token, is the class
                String s = dataValues.nextToken();
                if (dataValues.hasMoreTokens()) {
                    // Is data
                    trainData[i] = Double.parseDouble(s);
                    i++;
                } else {
                    trainClass = s;
                }
            }
            // Have data and class
            // Compute distance
            
        }
        
                
    }
  }

  public static class PredictClassReducer extends Reducer<Text,Text,Text,Text> {

    @Override
    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
    
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "KNN");
    job.setJarByClass(HD_KNN.class);
    job.setMapperClass(DistanceCalculatorMapper.class);
    job.setCombinerClass(PredictClassReducer.class);
    job.setReducerClass(PredictClassReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}