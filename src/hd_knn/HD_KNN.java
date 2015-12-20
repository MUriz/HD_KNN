/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hd_knn;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class HD_KNN {
    
    //put /Users/txumauriz/NetBeansProjects/HD_KNN/dist/HD_KNN.jar /home/hadoop/HD_KNN.jar
    public static class DistanceClassOutput implements WritableComparable<DistanceClassOutput> {
        public final Text instanceClass;
        public final DoubleWritable distance;
        
        public DistanceClassOutput() {
            instanceClass = new Text("-1");
            distance = new DoubleWritable(Double.MAX_VALUE);
        }

        public DistanceClassOutput(Text instanceClass, DoubleWritable distance) {
            this.instanceClass = instanceClass;
            this.distance = distance;
        }

        @Override
        public int compareTo(DistanceClassOutput o) {
            int classCmp = instanceClass.compareTo(o.instanceClass);
            if (classCmp != 0) {
                return classCmp;
            }
            return distance.compareTo(o.distance);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            DistanceClassOutput that = (DistanceClassOutput) o;

            if (instanceClass != null ? !instanceClass.equals(that.instanceClass) : that.instanceClass != null) return false;
            if (distance != null ? !distance.equals(that.distance) : that.distance != null) return false;

            return true;
        }

        @Override
        public int hashCode() {
            int result = instanceClass != null ? instanceClass.hashCode() : 0;
            result = 31 * result + (distance != null ? distance.hashCode() : 0);
            return result;
        }

        @Override
        public void write(DataOutput dataOutput) throws IOException {
            instanceClass.write(dataOutput);
            distance.write(dataOutput);
        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            instanceClass.readFields(dataInput);
            distance.readFields(dataInput);
        }
        
        @Override
        public String toString() {
            
            return instanceClass.toString() + " " + distance.toString();
            
        }
        
    }

    public static class DistanceCalculatorMapper extends Mapper<Object, Text, Text, DistanceClassOutput> {

        private Text emmitKey = new Text();
        
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
    
        private String[] tokenizeData(String data, boolean getClass) {

            data = data.trim();
            StringTokenizer str_tok = new StringTokenizer(data, ",");
            int n = str_tok.countTokens();
            if (!getClass) {
                n -= 1;
            }
            String[] ret = new String[n];
            for (int i = 0; i < n; i++) {
                ret[i] = str_tok.nextToken();
            }
            return ret;

        }

        private double euclideanDistance(String[] train, String[] test) {

            double s = 0;
            for (int i = 0; i < test.length; i++) {
                try {
                    double val1 = Double.valueOf(train[i]);
                    double val2 = Double.valueOf(test[i]);
                    s += (val1-val2)*(val1-val2);
                } catch (Exception e) {
                    
                }
                
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
                String[] testData = tokenizeData(nextTest, false);
                //Separate train data by "\n"
                StringTokenizer trainInstances = new StringTokenizer(value.toString(), "\n");
                while (trainInstances.hasMoreTokens()) {
                    String[] trainData = tokenizeData(trainInstances.nextToken(), true);
                    //Compute distance
                    DoubleWritable distance = new DoubleWritable(euclideanDistance(trainData, testData));
                    //tarin class: last value of trainData
                    Text trainClass = new Text(trainData[trainData.length - 1]);
                    // Emmit:
                    // key => test instance
                    // value => distance;class
                    DistanceClassOutput emmitValue = new DistanceClassOutput(trainClass, distance);
                    context.write(emmitKey, emmitValue);

                }

            }
        }
    }

    public static class PredictClassReducer extends Reducer<Text,DistanceClassOutput,Text,DistanceClassOutput> {

        @Override
        public void reduce(Text key, Iterable<DistanceClassOutput> values, Context context) throws IOException, InterruptedException {

            //IN:
            //  key => test instance
            //  value => array of distance;class
            //OUT;
            //  key => test instance
            //  value => class
            double minDistance = Double.MAX_VALUE;
            DistanceClassOutput emmitClass = new DistanceClassOutput();
            for (DistanceClassOutput val : values) {
                double distance = val.distance.get();
                if (distance < minDistance) {
                    minDistance = distance;
                    emmitClass = val;
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
    job.setOutputValueClass(DistanceClassOutput.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}