// MIDI Splitting program - to chop midi files into 30 second files
// Written by David Leung
// Last modified 29/08/2019

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Melanchall.DryWetMidi.Common;
using Melanchall.DryWetMidi.MusicTheory;
using Melanchall.DryWetMidi.Smf;
using Melanchall.DryWetMidi.Smf.Interaction;
using Melanchall.DryWetMidi.Tools;

namespace MIDISplitting
{
    class Program
    {
        static void Main(string[] args)
        {
            // Gets fullpath of each file in directory
            string[] filePaths = Directory.GetFiles(@"C:\\Users\\Dave\\Music\\FYPDataset\\2018", "*.mid", SearchOption.TopDirectoryOnly);   // 1. Edit directory here to change folder. Ensures only ".mid" files are returned
            int length = filePaths.Length;
            Console.WriteLine("Number of files: {0}", length);

            // Creates the grid to split midi files at (30 second intervals = 3E7)
            IGrid grid = new SteppedGrid(new MetricTimeSpan(30000000));


            // Reading and Split settings
            var readingSettings = new ReadingSettings
            {
                NotEnoughBytesPolicy = NotEnoughBytesPolicy.Abort,
                InvalidChunkSizePolicy = InvalidChunkSizePolicy.Abort
            };

            var splitSettings = new SliceMidiFileSettings
            {
                SplitNotes = true,
                PreserveTimes = false,
                PreserveTrackChunks = false
            };

            // Creating new output directory
            string newDirectory = "C:\\Users\\Dave\\Music\\FYPDataset\\2018\\Trimmed18";        // 2. Edit here to create new output folder
            Directory.CreateDirectory(newDirectory);

            int counterFiles = 0;
            // Begin loop for each file in directory
            foreach (string file in filePaths)
            {
                string filePath = file;
                Console.WriteLine(filePath);
                // Reads a singular midi file
                MidiFile mf = MidiFile.Read(filePath, readingSettings);

                // Creates a IEnumerable object of class MidiFile (containing all smaller midifiles)
                IEnumerable<MidiFile> midi = MidiFileSplitter.SplitByGrid(mf, grid, splitSettings).ToList();

                int counterSplits = 0;
                foreach (var i in midi)     // For each split file
                {
                    string fileWrite = String.Format("{0}\\SplitTrack{1}_{2}.MID", newDirectory, counterFiles, counterSplits);
                    Console.WriteLine(fileWrite);
                    i.Write(fileWrite,
                        overwriteFile: true,
                        format: MidiFileFormat.MultiTrack);
                    ++counterSplits;    // Increment split number
                }

                ++counterFiles;     // Increment file number
            }

            Console.WriteLine("Midi File Splitting Complete!");
            Console.Read();
        }
    }
}