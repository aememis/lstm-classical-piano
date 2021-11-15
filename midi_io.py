import glob
import numpy as np
import music21
import config as cfg
import pickle
from os.path import join, split
from keras.utils import np_utils
from sklearn.utils import shuffle
from numpy.core.multiarray import empty_like
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.backend import eval_in_eager_or_function, reshape

music21.environment.set('midiPath', cfg.BASE_FOLDER + '/MuseScore/MuseScore-3.5.0-x86_64.AppImage')
music21.environment.set('musicxmlPath', cfg.BASE_FOLDER + '/MuseScore/MuseScore-3.5.0-x86_64.AppImage')

notes_as_ints_global = None

len_set_notes = None
#Process every file in directory
def getData(targetPath):
	i = 0
	pieceBuffer = []
	files = glob.glob(targetPath + '/*.mid')[:30]
	#random.shuffle(files)
	for file in files:
		if len(file) < 4:
			print("NO MIDI FILE")
			exit()
		arrays = midiToArrays(file)
		if arrays is None: continue
		for matrice in arrays:
			pieceBuffer.append(matrice)
		i += 1
	
	print(str(i) + " midi files converted.")
	#print("input")
	#print("length of data :  ", len(pieceBuffer))
	#print("length of set  :  ", len(set(pieceBuffer)))
	#print((i, input_stats[i]) for i in input_stats)

	filepath = open(cfg.PATH["DATA"] + "/notes", "wb")
	pickle.dump(pieceBuffer, filepath)

	return pieceBuffer

#Create Input and Target Sequences
def createSeqInputs(data, x_seq_length, y_seq_length):
	data_int = []
	net_input = []
	net_output = []
	pitches_set = sorted(set(item for item in data))
	notes_as_ints = dict((note, number) for number, note in enumerate(pitches_set))
	#print(notes_as_ints)
	for s in data:
		data_int.append(notes_as_ints[s])

	pos = 0
	while pos + x_seq_length + y_seq_length < len(data_int):
		net_input.append(data_int[pos:pos + x_seq_length])
		net_output.append(data_int[pos + x_seq_length:pos + x_seq_length + y_seq_length])
		pos += 1

	net_input = np.reshape(net_input, (len(net_input), x_seq_length, 1))

	print ("net_input shape", net_input.shape)
	print ("net_output shape", np.array(net_output).shape)

	#print("net : --> ", net_input[:100])
	net_input_old = net_input
	net_input = net_input / float(len(pitches_set))
	net_output_old = net_output
	net_output = np_utils.to_categorical(net_output,num_classes=len(pitches_set), dtype=np.int)
	
	#net_output = np.reshape(net_output, (len(net_output), y_seq_length, 50))

	txtfile = open("summary.txt","w") 
	for i in range(len(net_input)):
		txtfile.write(str(net_input_old[i]) + "\n" + str(net_output_old[i]) + "\n" + str(net_output[i])) 
		txtfile.write('\n\n')
	txtfile.close()


	return net_input, net_output

output_stats = dict()
#Convert arrays to midi file
def arraysToMidi(inputs):
	notes = []
	offsetEnabled = False
	if cfg.GLOBAL["OFFSET"] == 1:
		offsetEnabled = True
	offset = 0
	for element in inputs:
		splitted = element.split(',')
		if offsetEnabled:
			offset = splitted[-1]
			splitted = splitted[:-1]
		#Note
		if len(splitted) == 1:
			newNote = music21.note.Note(splitted[0])
			newNote.offset = offset
			if not offsetEnabled:
				offset += 1
			#print("offset :  ", newNote.offset)
			notes.append(newNote)
			if element in output_stats: output_stats[splitted[0]] += 1
			else: output_stats[splitted[0]] = 1
		#Chord
		elif len(splitted) > 1:
			notes_in_chord = []
			for tone in splitted:
				newNote = music21.note.Note(tone)
				notes_in_chord.append(newNote)
				if tone in output_stats: output_stats[tone] += 1
				else: output_stats[tone] = 1
			newChord = music21.chord.Chord(notes_in_chord)
			newChord.offset = offset
			if not offsetEnabled:
				offset += 1
			#print("offset :  ", newChord.offset)
			notes.append(newChord)
		#Rest
		#elif rest:
		#
		#

	newScore = music21.stream.Score(id='LSTM')
	newPart = music21.stream.Part('Piano')
	newStream = music21.stream.Stream()
	for elem in notes:
		newStream.append(elem)
	newPart.append(newStream)
	newScore.insert(newPart)
	return newScore, output_stats

input_stats = dict()
#Convert midi files to arrays
def midiToArrays(midiPath):
	#Parse file
	score = music21.stream.Score()
	try:
		score = music21.converter.parse(midiPath)
		print("  o k  ", midiPath)
	except:
		print(" ERROR ", midiPath)
		return None

	notes_to_parse = None
	score = music21.converter.parse(midiPath)

	txtfile = open("flattened.txt","w") 
	for el in score.flat:
		txtfile.write(str(el) + "  " + str(el.offset))
		txtfile.write('\n')
	txtfile.close()

	#	try:  # file has instrument parts
	#		inst = music21.instrument.partitionByInstrument(score)
	#		#print("Number of instrument parts: " + str(len(inst.parts)))
	#		notes_to_parse = inst.parts[0].recurse()
	#	except:  # file has notes in a flat structure
	#		notes_to_parse = score.flat.notes

	arr = []
	prevOffset = 0
	offsetBuffer = sorted(set(el.offset for el in score.flat))
	for offset in offsetBuffer:
		elements_by_offset = score.flat.getElementsByOffset(offset)
		notes_by_offset = []
		for element in elements_by_offset:
			limit_chords = 3
			#Note
			if isinstance(element, music21.note.Note):
				notes_by_offset.append(str(element.pitch))
				if str(element.pitch) in input_stats: input_stats[str(element.pitch)] += 1
				else: input_stats[str(element.pitch)] = 1
			#Chord
			elif isinstance(element, music21.chord.Chord):
				i = 0
				for tone in element:
					if i == limit_chords:
						break
					i += 1
					notes_by_offset.append(str(tone.pitch))
					if str(tone.pitch) in input_stats: input_stats[str(tone.pitch)] += 1
					else: input_stats[str(tone.pitch)] = 1
			#Rest
			#elif isinstance(element, music21.note.Rest):
			#
			#
		if len(notes_by_offset) == 0:
			continue

		notes_classes = [i[-1] for i in notes_by_offset]
		notes_by_offset = [x for _,x in sorted(zip(notes_classes, notes_by_offset))]
		notes_string = notes_by_offset[-1]
		#print(notes_string)
		if cfg.GLOBAL["OFFSET"] == 1:
			notes_string = notes_string + ',' + str(offset - prevOffset)

		"""
		if len(notes_by_offset) > 1:
			notes_string = ','.join(sorted(set(s for s in notes_by_offset)))
			if cfg.GLOBAL["OFFSET"] == 1:
				notes_string = notes_string + ',' + str(offset - prevOffset)
		else:
			notes_string = notes_by_offset[0]
			if cfg.GLOBAL["OFFSET"] == 1:
				notes_string = notes_string + ',' + str(offset - prevOffset)
		"""
		if cfg.GLOBAL["OFFSET"] == 1:
			prevOffset = offset

		#print("arr: \n", arr[:100])

		arr.append(notes_string)

	#print("length of arr :  ", len(arr))
	#print("length of set :  ", len(set(arr)))
	#a,b = arraysToMidi(arr)
	#a.show('midi')

	#print(arr[:100])
	#input()

	return arr

def compile_stats(arr):
	stat_array = np.zeros(12)
	for i in range(12):
		stat_array[i] = arr[i] + arr[i + 12]
	return stat_array




	# ux minimal
	# product designer
	# paraşüt
	# ui engineer
	# design system
	# nn group
	# oktay