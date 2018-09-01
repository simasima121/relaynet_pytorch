def dice_met(inp):
	first = inp.replace('Average Scores [ ','')
	split = first.split()
	string = '&'
	for i in range(0,len(split)):

		if split[i]!= ']':
			if split[i] == 'nan':
				val = 'nan'
			else:
				value = split[i].replace(']','')
				val = float(value)
				new_val = round(val,2)
			string += str(new_val) + '&'
	string = string[:-1]
	print(string+'\\\\')

def thickness_met(inp):
	first = inp.replace('Average Scores [ ','')
	split = first.split()
	string = '&'
	for i in range(0,len(split)):

		if split[i] != ']':
			value = split[i].replace(']','')
			val = float(value)
			new_val = int(round(val,0))
			string += str(new_val) + '&'
	string = string[:-1]
	print(string+'\\\\')

x = "Average Scores [ 0.78  0.72  0.42  0.59  0.55  0.66   nan]"
y = "Average Scores [ 141.64   24.72   94.95   37.49   10.42   12.36    7.43]"
z = "Average Scores [ 56.93  10.76  87.69  27.54  10.04  10.69   7.43]"


dice_met(x)
thickness_met(y)
thickness_met(z)
