use proc_hacks::type_of;

fn main() {
    let x = 5;
    type Foo = type_of!(x);
}
