let x = 10;

// simulate do..while using loop
loop {
    print(x);
    x = x - 1;
	if x <= 0 { break; }
}

let a = loop {
    break 5;
}

print(a);