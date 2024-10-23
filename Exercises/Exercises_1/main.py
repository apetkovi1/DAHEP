import ClassDef

Z = ClassDef.Boson("Z boson", 1, 70)
Z.PrintInfo()

H = ClassDef.Higgs("Higgs boson",0,60,125)
print(f"Higgs boson energy is {H.Energy()} GeV")