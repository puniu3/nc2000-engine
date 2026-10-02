use nc2000_engine::state::{Battle, DataBag, EffectState, Pokemon, Scalar, Side, Status};
fn main() {
    println!(
        "{}",
        serde_json::json!({
            "Status": size_of::<Status>(), "Scalar": size_of::<Scalar>(),
            "DataBag": size_of::<DataBag>(), "EffectState": size_of::<EffectState>(),
            "Pokemon": size_of::<Pokemon>(), "Side": size_of::<Side>(), "Battle": size_of::<Battle>(),
        })
    );
}
