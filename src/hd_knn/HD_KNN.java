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

    public static class DistanceCalculatorMapper extends Mapper<Object, Text, Text, Text> {

        private final Text emmitKey = new Text();
        private final Text emmitValue = new Text();
        
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
    
        private String[] tokenizeData(String data) {

            data = data.trim();
            StringTokenizer str_tok = new StringTokenizer(data, ",");
            String[] ret = new String[str_tok.countTokens()];
            int i = 0;
            while (str_tok.hasMoreTokens()) {
                ret[i] = str_tok.nextToken();
                i++;
            }
            return ret;

        }

        private double euclideanDistance(String[] train, String[] test) {

            double s = 0;
            for (int i = 0; i < test.length; i++) {
                double val1 = Double.valueOf(train[i]);
                double val2 = Double.valueOf(test[i]);
                s += (val1-val2)*(val1-val2);
            }
            return Math.sqrt(s);

        }
      
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            // In value train instances
            // x11,x12,...,x1m
            // x21,x22,...,x2m
            // ...............
            // xn1,xn2,...,xnm
            // Read test data
            String testDataCSV = readTest(context.getConfiguration());
            //Separa test data by "\n"
            StringTokenizer testInstances = new StringTokenizer(testDataCSV, "\n");
            while (testInstances.hasMoreTokens()) {
                String nextTest = testInstances.nextToken();
                emmitKey.set(nextTest);
                String[] testData = tokenizeData(nextTest);
                //Separate train data by "\n"
                StringTokenizer trainInstances = new StringTokenizer(value.toString(), "\n");
                while (trainInstances.hasMoreTokens()) {
                    String[] trainData = tokenizeData(trainInstances.nextToken());
                    //Compute distance
                    double distance = euclideanDistance(trainData, testData);
                    //tarin class: last value of trainData
                    String trainClass = trainData[trainData.length - 1];
                    // Emmit:
                    // key => test instance
                    // value => distance;class
                    emmitValue.set(String.valueOf(distance) + ";" + trainClass);
                    context.write(emmitKey, emmitValue);

                }

            }
        }
    }

    public static class PredictClassReducer extends Reducer<Text,Text,Text,Text> {
      
        private final Text emmitClass = new Text();

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            //IN:
            //  key => test instance
            //  value => array of distance;class
            //OUT;
            //  key => test instance
            //  value => class
            emmitClass.set("-1");
            double minDistance = Double.MAX_VALUE;
            for (Text val : values) {
                // Split ";" and get distance and value
                StringTokenizer str_tok = new StringTokenizer(val.toString(), ";");
                double distance = Double.parseDouble(str_tok.nextToken());
                if (distance < minDistance) {
                    minDistance = distance;
                    emmitClass.set(str_tok.nextToken());
                }
            }

            context.write(key, emmitClass);
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