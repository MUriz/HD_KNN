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
        
        public DistanceClassOutput(DistanceClassOutput o) {
            this.instanceClass = new Text(o.instanceClass);
            this.distance = new DoubleWritable(o.distance.get());
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
        
        private String readTest(Configuration conf) {

            try{
                String test_file = conf.get("test_file");
                Path pt = new Path(test_file);
                FileSystem fs = FileSystem.get( new URI(test_file), conf);
                LocalFileSystem localFileSystem = fs.getLocal(conf);
                BufferedReader bufferRedaer = new BufferedReader(new InputStreamReader(localFileSystem.open(pt)));

                String str = null;
                StringBuilder str_build = new StringBuilder();
                while ((str = bufferRedaer.readLine())!= null)
                {
                    str_build.append(str+"\n");
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
      
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            // In value train instances
            // x11,x12,...,x1m
            // x21,x22,...,x2m
            // ...............
            // xn1,xn2,...,xnm
            String dist = context.getConfiguration().get("distance", "0");
            KNN_distance knn_distance;
            switch (dist) {
                case "0":
                    knn_distance = new KNN_EuclideanDistance();
                    break;
                case "1":
                    knn_distance = new KNN_ManhattanDistance();
                    break;
                case "2":
                    knn_distance = new KNN_ChebyshevDistance();
                    break;
                default:
                    knn_distance = new KNN_EuclideanDistance();
                    break;
            }
            // Read test data
            String testDataCSV = readTest(context.getConfiguration());
            //Separa test data by "\n"
            StringTokenizer testInstances = new StringTokenizer(testDataCSV, "\n");
            while (testInstances.hasMoreTokens()) {
                String nextTest = testInstances.nextToken();
                Text emmitKey = new Text(nextTest);
                String[] testData = tokenizeData(nextTest, false);
                //Separate train data by "\n"
                StringTokenizer trainInstances = new StringTokenizer(value.toString(), "\n");
                while (trainInstances.hasMoreTokens()) {
                    String[] trainData = tokenizeData(trainInstances.nextToken(), true);
                    //Compute distance
                    DoubleWritable distance = new DoubleWritable(knn_distance.distance(trainData, testData));
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
            KNN_method knn_method;
            String meth = context.getConfiguration().get("knn_method", "0");
            switch (meth) {
                case "0":
                    knn_method = new KNN_Normal();
                    break;
                case "1":
                    knn_method = new KNN_MediaVecinos();
                    break;
                case "2":
                    knn_method = new KNN_VotoInverso();
                    break;
                default:
                    knn_method = new KNN_Normal();
                    break;
            }
            int k = context.getConfiguration().getInt("k", 1);
            DistanceClassOutput[] nearest = new DistanceClassOutput[k];
            for (int i = 0; i < k; i++) {
                nearest[i] = new DistanceClassOutput(new Text("-1"), new DoubleWritable(Double.MAX_VALUE));
            }
            for (DistanceClassOutput val : values) {
                nearest = update(nearest, val);
            }

            DistanceClassOutput emmitClass = new DistanceClassOutput(knn_method.getReducerOutput(nearest));

            context.write(key, emmitClass);
        }
        
        public DistanceClassOutput[] update(DistanceClassOutput[] nearest, DistanceClassOutput current) {
            
            DistanceClassOutput[] ret = new DistanceClassOutput[nearest.length];
            for (int i = 0; i < nearest.length; i++) {
                ret[i] = new DistanceClassOutput(nearest[i]);
            }
            
            double max = nearest[0].distance.get();
            int i_max = 0;
            for (int i = 1; i < nearest.length; i++) {
                if (nearest[i].distance.get() > max) {
                    max = nearest[i].distance.get();
                    i_max = i;
                }
            }
            
            if (current.distance.get() < max) {
                ret[i_max] = new DistanceClassOutput(current);
            }
            
            return ret;
            
        }
        
    }

    public static void main(String[] args) throws Exception {
        
        // argumentos
        // Variante KNN: 0 Normal, 1 Media, 2 Inversa de la distancia
        // Distancia a utilizar: 0 Euclidea, 1 Manhattan, 2 Chebyshev
        // k
        // test_file
        // Input path
        // Output path
        if (args.length != 6) {
            System.out.println("Arguments: knn_type distance k test_file input_path output_path");
            System.exit(-1);
        }
        Configuration conf = new Configuration();
        conf.set("knn_method", args[0]);
        conf.set("distance", args[1]);
        conf.setInt("k", Integer.parseInt(args[2]));
        conf.set("test_file", args[3]);
        Job job = Job.getInstance(conf, "KNN");
        job.setJarByClass(HD_KNN.class);
        job.setMapperClass(DistanceCalculatorMapper.class);
        job.setCombinerClass(PredictClassReducer.class);
        job.setReducerClass(PredictClassReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DistanceClassOutput.class);
        FileInputFormat.addInputPath(job, new Path(args[4]));
        FileOutputFormat.setOutputPath(job, new Path(args[5]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}