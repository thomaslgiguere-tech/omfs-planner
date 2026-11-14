-- DentalSeg : glisse un dossier DICOM (ou un .dcm, .nrrd, .nii.gz) sur l'icône.
property pythonPath : "~/miniconda3/envs/dentalseg/bin/python"
property scriptPath : "~/Projects/omfs-planner/segmentation/segment.py"

on open droppedItems
	set args to ""
	repeat with anItem in droppedItems
		set args to args & " " & quoted form of POSIX path of anItem
	end repeat
	runSegmentation(args)
end open

on run
	set choice to button returned of (display dialog "Glisse un dossier DICOM ou un fichier de scan sur l'icône DentalSeg, ou choisis-le ici." buttons {"Annuler", "Fichier…", "Dossier DICOM…"} default button 3 with title "DentalSeg")
	if choice is "Dossier DICOM…" then
		set picked to choose folder with prompt "Dossier DICOM à segmenter :"
	else if choice is "Fichier…" then
		set picked to choose file with prompt "Scan à segmenter (.dcm, .nrrd, .nii.gz) :"
	else
		return
	end if
	runSegmentation(" " & quoted form of POSIX path of picked)
end run

on runSegmentation(args)
	set mode to button returned of (display dialog "Mode de segmentation :" & return & return & "Précis : le volume vu 8 fois en miroir, environ 30 min." & return & "Rapide : un seul passage, environ 3 min." & return & return & "Les probabilités (surfaces sous le voxel) sont gardées dans les deux cas." buttons {"Annuler", "Rapide", "Précis"} default button 3 cancel button 1 with title "DentalSeg")
	if mode is "Rapide" then set args to " --rapide" & args
	tell application "Terminal"
		activate
		do script "clear; " & pythonPath & " " & scriptPath & args & "; echo; echo 'Tu peux fermer cette fenêtre.'"
	end tell
end runSegmentation
