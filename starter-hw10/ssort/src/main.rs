
extern crate rand;
use rand::Rng;

use std::env;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::fs::{File, OpenOptions};
use std::f32;
use std::thread;
use std::sync::{Arc, Barrier, Mutex};
use bytes::Buf;  

fn main() {
    let args : Vec<String> = env::args().collect();
    if args.len() != 4 {
        println!("Usage: {} <threads> input output", args[0]);
    }

    let threads =  args[1].parse::<usize>().unwrap();
    let inp_path = &args[2];
    let out_path = &args[3];

    // Calculate pivots
    let mut inpf = File::open(inp_path).unwrap();
    let pivots = find_pivots(&mut inpf, threads);
    let size = read_size(&mut inpf);
    let num_floats = read_num_floats(&mut inpf);

    // Create output file
    {
        let mut outf = File::create(out_path).unwrap();
        let tmp_floats = num_floats.to_ne_bytes();
        outf.write_all(&tmp_floats).unwrap();
        outf.set_len(size).unwrap();
    }

    let mut workers = vec![];
    
    // Spawn worker threads
    let sizes = Arc::new(Mutex::new(vec![0u64; threads]));
    let barrier = Arc::new(Barrier::new(threads));

    for ii in 0..threads {
        let inp = inp_path.clone();
        let out = out_path.clone();
        let piv = pivots.clone();
        let szs = sizes.clone();
        let bar = barrier.clone();

        let tt = thread::spawn(move || {
            worker(ii, inp, out, piv, szs, bar);
        });
        workers.push(tt);
    }

    // Join worker threads
    for tt in workers {
        tt.join().unwrap();
    }
}

fn read_size(file: &mut File) -> u64 {
    // Read size field from data file
    file.metadata().unwrap().len()
}

// Returns the number of floats that a file contains
// it is an arithmetic on size of file and not a generic api
fn read_num_floats(file: &mut File) -> u64 {
    // Read size field from data file
    let size_bytes_floats = file.metadata().unwrap().len() - 8;
    let num_floats = size_bytes_floats / 4;
    num_floats
}


fn read_item(file: &mut File, ii: u64) -> f32 {
    // Read the ii'th float from data file
    // Perform arithmetic by below equation
    // to get bytes offset of float index
    let float_ofs = 8 + ii*4;
    // Seek the byte or put the cursor at that
    // byte offset and unwrap it to get the value
    file.seek(SeekFrom::Start(float_ofs)).unwrap();
    let mut tmp = [0u8; 4];
    // Read 4 bytes which is a float
    file.read_exact(&mut tmp).unwrap();
    let xx = Cursor::new(tmp).get_f32_le();
    //file.seek(SeekFrom::Start(0)).unwrap(); 
    xx
}

fn sample(file: &mut File, count: usize, size: u64) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut ys = vec![];
    let count_u64 = count as u64;

    let factor_range = size / count_u64 ; 
    //  Sample 'count' random items from the
    // provided file

    // Start with a lower bound and increment it by a factor
    // as calculated above , after generating a random number
    // in that range . 
    let lower_bound = 0;
    let upper_bound = factor_range * count_u64;
    let mut size_iterator = lower_bound;
    let mut done = false;
    while !done {
      let range_lower     = size_iterator;
      let range_upper     = size_iterator + factor_range; 
      let random_index    = rng.gen_range(range_lower,range_upper);
      let value_rnd_index = read_item(file,random_index); 
      ys.push(value_rnd_index);
      size_iterator = size_iterator + factor_range;
      if size_iterator == upper_bound 
      {
        done = true;    
      }
    }

    ys
}

fn find_pivots(file: &mut File, threads: usize) -> Vec<f32> {
    // Sample 3*(threads-1) items from the file
    let count_samples = 3*(threads-1);
    let num_floats    = read_num_floats(file);

    let mut pivots = vec![0f32];

    // For threads = 1 , the pivot should be 0.0 as a lower limit
    // so that it can be handled later in worker code
    if count_samples == 0
    {
      // do nothing
    }
    else
    {

      let mut samples = sample(file,count_samples,num_floats);

      // Sort the sampled list . ( attr : picked up below line from rust cookbook )
      samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

      // Pivots are nothing but medians of samples array , that would be
      // pushed onto pivots array as below . 
      let mut done = false;
      let lower_bound = 0;
      let upper_bound = count_samples;
      let mut samples_iterator = lower_bound;

      while !done
      {
        let sum = samples[samples_iterator] + samples[samples_iterator+1]+samples[samples_iterator+2];
        let median = sum / 3.0;
        pivots.push(median);
        samples_iterator = samples_iterator + 3;
        if samples_iterator == upper_bound 
        {
          done = true;
        }
      }
    }

    pivots
}

fn worker(tid: usize, inp_path: String, out_path: String, pivots: Vec<f32>,
          sizes: Arc<Mutex<Vec<u64>>>, bb: Arc<Barrier>) {

    // Open input as local fh
    let mut inpf = File::open(inp_path).unwrap();
    let last_index = pivots.len() - 1;

    let lower_limit = pivots[tid];
    let mut upper_limit = 1000000.00;

    if tid < last_index
    {
      upper_limit = pivots[tid+1];    
    }

    // Scan to collect local data
    let mut data = vec![];
   
    let num_floats = read_num_floats(&mut inpf);
    
    // Iterator over the floats in file and create a local
    // array
    for iterator_floats in 0..num_floats
    {
       let data_item = read_item(&mut inpf,iterator_floats);
       if (data_item >= lower_limit) && (data_item < upper_limit)
       {
          data.push(data_item);     
       }
    }

    //  Write local size to shared sizes
    {
       // curly braces to scope our lock guard
       let mut sizes_ref = sizes.lock().unwrap();
       sizes_ref[tid] = data.len() as u64;
    }

    //  Sort local data
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Here's our printout
    println!("{}: start {}, count {}", tid, &data[0], data.len());

    // Write data to local buffer
    let mut cur = Cursor::new(vec![]);

    for xx in &data {
        let tmp = xx.to_bits().to_ne_bytes();
        cur.write_all(&tmp).unwrap();
    }

    bb.wait();

    let mut file_seek_offset = 8;

    {
      // curly braces to scope our lock guard
      let sizes_ref = sizes.lock().unwrap();
 
      for tid_iter in 0..tid
      {
        file_seek_offset = file_seek_offset + sizes_ref[tid_iter] * 4;  
      }
    }

    let mut outf = OpenOptions::new()
        .read(true)
        .write(true)
        .open(out_path).unwrap();

    // Seek and write local buffer.
    outf.seek(SeekFrom::Start(file_seek_offset)).unwrap(); 
    outf.write(cur.get_ref()).unwrap();
}
