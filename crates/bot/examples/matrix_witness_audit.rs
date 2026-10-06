use nc2000_bot::exact::solve_matrix_full;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};

fn main() {
    let args:Vec<_>=std::env::args().collect();
    assert_eq!(args.len(),3,"input output");
    let raw=std::fs::read(&args[1]).unwrap();
    let input:Value=serde_json::from_slice(&raw).unwrap();
    let mut total=0;
    let mut invalid=0;
    let mut records=Vec::new();
    for root in input["rows"].as_array().unwrap() {
        let dimensions=root["model"]["dimensions"].as_array().unwrap();
        let rows=dimensions[0].as_u64().unwrap() as usize;
        let cols=dimensions[1].as_u64().unwrap() as usize;
        assert!(rows<16 && cols<16);
        let cells=root["model"]["cell_bounds"].as_array().unwrap();
        let mut worst=0.;
        let mut record=None;
        let mut count=0;
        for endpoint in 0..2 {
            for rm in 1usize..(1<<rows) {
                if rm.count_ones()<2 {continue;}
                let ri:Vec<_>=(0..rows).filter(|&i|rm&(1<<i)!=0).collect();
                for cm in 1usize..(1<<cols) {
                    if cm.count_ones()<2 {continue;}
                    let cj:Vec<_>=(0..cols).filter(|&j|cm&(1<<j)!=0).collect();
                    let matrix:Vec<_>=ri.iter().flat_map(|&i|cj.iter().map(move |&j|cells[i*cols+j][endpoint].as_f64().unwrap())).collect();
                    let solution=solve_matrix_full(&matrix,ri.len(),cj.len());
                    assert!(solution.x.iter().chain(&solution.y).all(|p|p.is_finite()));
                    let minimum=solution.x.iter().chain(&solution.y).copied().fold(0.,f64::min);
                    total+=1;
                    if minimum<0. {
                        invalid+=1;count+=1;
                        if minimum<worst {
                            worst=minimum;
                            record=Some(json!({"game":root["game"],"decision":root["decision"],"endpoint":endpoint,
                                "rows":ri,"cols":cj,"matrix":matrix,"x":solution.x,"y":solution.y,
                                "minimum_weight":minimum,"value":solution.value,"gap":solution.gap}));
                        }
                    }
                }
            }
        }
        if let Some(mut record)=record {
            record["invalid_submatrices"]=json!(count);
            records.push(record);
        }
    }
    let sha=|raw:&[u8]|format!("{:x}",Sha256::digest(raw));
    let source_hashes=["crates/bot/src/exact.rs","crates/bot/examples/matrix_witness_audit.rs"].map(|p|(p,sha(&std::fs::read(p).unwrap())));
    let result=json!({"matrices":total,"invalid":invalid,"records":records,"input_sha256":sha(&raw),
        "source_hashes":source_hashes});
    let mut output=std::fs::File::create_new(&args[2]).unwrap();
    serde_json::to_writer_pretty(&mut output,&result).unwrap();
}
