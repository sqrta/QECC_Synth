output := OutputTextFile( "result", false);;
filedir:=DirectoriesPackageLibrary("QDistRnd","matrices");;
lisX:=ReadMTXE("66Hx.mtx",0);;
GX:=lisX[3];;
lisZ:=ReadMTXE("66Hz.mtx",0);;
GZ:=lisZ[3];;
d:=DistRandCSS(GX,GZ,100,1,2:field:=GF(2));
AppendTo(output, d);
CloseStream(output);