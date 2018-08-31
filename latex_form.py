def dice_met(inp):
	first = inp.replace('Average Scores [ ','')
	split = first.split()

	string = '& '
	for i in range(1,len(split)):

		if split[i]!= ']':
			string += split[i] + ' & '
	print(string[:-2]+'\\\\')

def thickness_met(inp):
	first = inp.replace('Average Scores [ ','')
	split = first.split()

	string = '& '
	for i in range(1,len(split)):

		if split[i]!= ']':
			val = float(split[i])
			new_val = int(round(val,0))
			string += str(new_val) + ' & '
	print(string[:-2]+'\\\\')

x = "Average Scores [ 0.316  0.01   0.044  0.071  0.089  0.097  0.082  0.03 ]"
y = "Average Scores [ 179.839   10.813   25.374   20.678   17.265   36.794    8.95     7.3  ]"
z = "Average Scores [  96.293  123.936   25.761   86.402   28.225   50.032   12.533   10.734 ]"

dice_met(x)
thickness_met(y)
thickness_met(z)
