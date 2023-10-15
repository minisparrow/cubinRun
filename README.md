# cubinRun 

## Introduction 

A simple demo for cuda source compile to runtime 

## Flow 

(Compile Time): [cuda kernel] -> compileCUDAtoPTX -> [PTX] -> compilePTXtoCUBIN -> [cubin] 

(Runtime): launchCUBIN 
