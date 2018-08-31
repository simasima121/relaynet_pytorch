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

x = "Average Scores [ 0.    0.02  0.08  0.02   nan   nan  0.01]"
y = "Average Scores [  15.17   35.44   24.8     8.76  112.79    4.5     3.24]"
z = "Average Scores [ 157.08   25.14   94.23   21.5   100.82   11.1     7.62]"


dice_met(x)
thickness_met(y)
thickness_met(z)
