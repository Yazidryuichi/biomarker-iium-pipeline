




/***************** 
 * Psychopy *
 *****************/

import { core, data, sound, util, visual, hardware } from './lib/psychojs-2025.2.4.js';
const { PsychoJS } = core;
const { TrialHandler, MultiStairHandler } = data;
const { Scheduler } = util;
//some handy aliases as in the psychopy scripts;
const { abs, sin, cos, PI: pi, sqrt } = Math;
const { round } = util;


// store info about the experiment session:
let expName = 'psychopy';  // from the Builder filename that created this script
let expInfo = {
    'participant': f'{randint(0, 999999):06.0f}',
    'session': '001',
};
let PILOTING = util.getUrlParameters().has('__pilotToken');

// Start code blocks for 'Before Experiment'
// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([0,0,0]),
  units: 'height',
  waitBlanking: true,
  backgroundImage: '',
  backgroundFit: 'none',
});
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); },flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(instruksiRoutineBegin());
flowScheduler.add(instruksiRoutineEachFrame());
flowScheduler.add(instruksiRoutineEnd());
const practice_loopLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(practice_loopLoopBegin(practice_loopLoopScheduler));
flowScheduler.add(practice_loopLoopScheduler);
flowScheduler.add(practice_loopLoopEnd);



flowScheduler.add(readyRoutineBegin());
flowScheduler.add(readyRoutineEachFrame());
flowScheduler.add(readyRoutineEnd());
const main_loopLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(main_loopLoopBegin(main_loopLoopScheduler));
flowScheduler.add(main_loopLoopScheduler);
flowScheduler.add(main_loopLoopEnd);


flowScheduler.add(break_3RoutineBegin());
flowScheduler.add(break_3RoutineEachFrame());
flowScheduler.add(break_3RoutineEnd());
const main_loop2LoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(main_loop2LoopBegin(main_loop2LoopScheduler));
flowScheduler.add(main_loop2LoopScheduler);
flowScheduler.add(main_loop2LoopEnd);


flowScheduler.add(endRoutineBegin());
flowScheduler.add(endRoutineEachFrame());
flowScheduler.add(endRoutineEnd());
flowScheduler.add(quitPsychoJS, 'Thank you for your patience.', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, 'Thank you for your patience.', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    // resources:
    {'name': 'conditions.xlsx', 'path': 'conditions.xlsx'},
    {'name': 'congruent_left.png', 'path': 'congruent_left.png'},
    {'name': 'congruent_right.png', 'path': 'congruent_right.png'},
    {'name': 'incongruent_left.png', 'path': 'incongruent_left.png'},
    {'name': 'conditions.xlsx', 'path': 'conditions.xlsx'},
    {'name': 'congruent_left.png', 'path': 'congruent_left.png'},
    {'name': 'congruent_right.png', 'path': 'congruent_right.png'},
    {'name': 'incongruent_left.png', 'path': 'incongruent_left.png'},
    {'name': 'conditions.xlsx', 'path': 'conditions.xlsx'},
    {'name': 'congruent_left.png', 'path': 'congruent_left.png'},
    {'name': 'congruent_right.png', 'path': 'congruent_right.png'},
    {'name': 'incongruent_left.png', 'path': 'incongruent_left.png'},
    {'name': 'default.png', 'path': 'https://pavlovia.org/assets/default/default.png'},
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.INFO);

async function updateInfo() {
  currentLoop = psychoJS.experiment;  // right now there are no loops
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2025.2.4';
  expInfo['OS'] = window.navigator.platform;


  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  

  
  psychoJS.experiment.dataFileName = (("." + "/") + (u'data/%s_%s_%s' % [expInfo['participant'], expName, expInfo['date']]));
  psychoJS.experiment.field_separator = '\t';


  return Scheduler.Event.NEXT;
}

async function experimentInit() {
  // Initialize components for Routine "instruksi"
  instruksiClock = new util.Clock();
  instr_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'instr_text',
    text: 'Kamu akan melihat sekumpulan ikan di layar.\n    \n    Tugasmu: lihat ikan yang ada di TENGAH.\n    \n    Ikan tengah menghadap KIRI → tekan tombol KIRI\n    Ikan tengah menghadap KANAN → tekan tombol KANAN\n    \n    Abaikan ikan-ikan yang lain!\n    \n    Tekan SPASI untuk mulai latihan.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: False, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  instr_key = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "trial"
  trialClock = new util.Clock();
  fixation = new visual.TextStim({
    win: psychoJS.window,
    name: 'fixation',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: False, height: 0.1,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  response = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  fish_stim = new visual.ImageStim({
    win : psychoJS.window,
    name : 'fish_stim', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0, 0], 
    draggable: False,
    size : [0.9, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  // Initialize components for Routine "feedback"
  feedbackClock = new util.Clock();
  feedback_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'feedback_text',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: False, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Initialize components for Routine "ready"
  readyClock = new util.Clock();
  ready_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'ready_text',
    text: 'Latihan selesai!\n\nSekarang tes yang sebenarnya.\n\nTekan SPASI untuk mulai.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: False, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  ready_key = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "break_3"
  break_3Clock = new util.Clock();
  break_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'break_text',
    text: 'Istirahat sebentar.\nTekan SPASI untuk lanjut ke bagian 2.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: False, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  break_key = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "end"
  endClock = new util.Clock();
  end_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'end_text',
    text: 'Selesai!\nTerima kasih sudah berpartisipasi.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: False, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}

function instruksiRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'instruksi' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    instruksiClock.reset();
    routineTimer.reset();
    instruksiMaxDurationReached = false;
    // update component parameters for each repeat
    instr_key.keys = undefined;
    instr_key.rt = undefined;
    _instr_key_allKeys = [];
    psychoJS.experiment.addData('instruksi.started', globalClock.getTime());
    instruksiMaxDuration = None
    // keep track of which components have finished
    instruksiComponents = [];
    instruksiComponents.push(instr_text);
    instruksiComponents.push(instr_key);
    
    for (const thisComponent of instruksiComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}

function instruksiRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'instruksi' ---
    // get current time
    t = instruksiClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instr_text* updates
    if (t >= 0.0 && instr_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instr_text.tStart = t;  // (not accounting for frame time here)
      instr_text.frameNStart = frameN;  // exact frame index
      
      instr_text.setAutoDraw(true);
    }
    
    
    // if instr_text is active this frame...
    if (instr_text.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *instr_key* updates
    if (t >= 0.0 && instr_key.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instr_key.tStart = t;  // (not accounting for frame time here)
      instr_key.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { instr_key.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { instr_key.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { instr_key.clearEvents(); });
    }
    
    // if instr_key is active this frame...
    if (instr_key.status === PsychoJS.Status.STARTED) {
      let theseKeys = instr_key.getKeys({
        keyList: typeof 'space' === 'string' ? ['space'] : 'space', 
        waitRelease: false
      });
      _instr_key_allKeys = _instr_key_allKeys.concat(theseKeys);
      if (_instr_key_allKeys.length > 0) {
        instr_key.keys = _instr_key_allKeys[_instr_key_allKeys.length - 1].name;  // just the last key pressed
        instr_key.rt = _instr_key_allKeys[_instr_key_allKeys.length - 1].rt;
        instr_key.duration = _instr_key_allKeys[_instr_key_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      routineForceEnded = true;
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruksiComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function instruksiRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'instruksi' ---
    for (const thisComponent of instruksiComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('instruksi.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(instr_key.corr, level);
    }
    psychoJS.experiment.addData('instr_key.keys', instr_key.keys);
    if (typeof instr_key.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('instr_key.rt', instr_key.rt);
        psychoJS.experiment.addData('instr_key.duration', instr_key.duration);
        routineTimer.reset();
        }
    
    instr_key.stop();
    // the Routine "instruksi" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function practice_loopLoopBegin(practice_loopLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    practice_loop = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 1, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'conditions.xlsx',
      seed: undefined, name: 'practice_loop'
    });
    psychoJS.experiment.addLoop(practice_loop); // add the loop to the experiment
    currentLoop = practice_loop;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisPractice_loop of practice_loop) {
      snapshot = practice_loop.getSnapshot();
      practice_loopLoopScheduler.add(importConditions(snapshot));
      practice_loopLoopScheduler.add(trialRoutineBegin(snapshot));
      practice_loopLoopScheduler.add(trialRoutineEachFrame());
      practice_loopLoopScheduler.add(trialRoutineEnd(snapshot));
      practice_loopLoopScheduler.add(feedbackRoutineBegin(snapshot));
      practice_loopLoopScheduler.add(feedbackRoutineEachFrame());
      practice_loopLoopScheduler.add(feedbackRoutineEnd(snapshot));
      practice_loopLoopScheduler.add(practice_loopLoopEndIteration(practice_loopLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}

async function practice_loopLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(practice_loop);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function practice_loopLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}

function main_loopLoopBegin(main_loopLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    main_loop = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 1, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'conditions.xlsx',
      seed: undefined, name: 'main_loop'
    });
    psychoJS.experiment.addLoop(main_loop); // add the loop to the experiment
    currentLoop = main_loop;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisMain_loop of main_loop) {
      snapshot = main_loop.getSnapshot();
      main_loopLoopScheduler.add(importConditions(snapshot));
      main_loopLoopScheduler.add(trialRoutineBegin(snapshot));
      main_loopLoopScheduler.add(trialRoutineEachFrame());
      main_loopLoopScheduler.add(trialRoutineEnd(snapshot));
      main_loopLoopScheduler.add(main_loopLoopEndIteration(main_loopLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}

async function main_loopLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(main_loop);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function main_loopLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}

function main_loop2LoopBegin(main_loop2LoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    main_loop2 = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 1, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'conditions.xlsx',
      seed: undefined, name: 'main_loop2'
    });
    psychoJS.experiment.addLoop(main_loop2); // add the loop to the experiment
    currentLoop = main_loop2;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisMain_loop2 of main_loop2) {
      snapshot = main_loop2.getSnapshot();
      main_loop2LoopScheduler.add(importConditions(snapshot));
      main_loop2LoopScheduler.add(trialRoutineBegin(snapshot));
      main_loop2LoopScheduler.add(trialRoutineEachFrame());
      main_loop2LoopScheduler.add(trialRoutineEnd(snapshot));
      main_loop2LoopScheduler.add(main_loop2LoopEndIteration(main_loop2LoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}

async function main_loop2LoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(main_loop2);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function main_loop2LoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}

function trialRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'trial' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    trialClock.reset();
    routineTimer.reset();
    trialMaxDurationReached = false;
    // update component parameters for each repeat
    response.keys = undefined;
    response.rt = undefined;
    _response_allKeys = [];
    fish_stim.setImage(image_file);
    psychoJS.experiment.addData('trial.started', globalClock.getTime());
    trialMaxDuration = None
    // keep track of which components have finished
    trialComponents = [];
    trialComponents.push(fixation);
    trialComponents.push(response);
    trialComponents.push(fish_stim);
    
    for (const thisComponent of trialComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}

function trialRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'trial' ---
    // get current time
    t = trialClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *fixation* updates
    if (t >= 0.0 && fixation.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      fixation.tStart = t;  // (not accounting for frame time here)
      fixation.frameNStart = frameN;  // exact frame index
      
      fixation.setAutoDraw(true);
    }
    
    
    // if fixation is active this frame...
    if (fixation.status === PsychoJS.Status.STARTED) {
    }
    
    frameRemains = 0.0 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (fixation.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      // keep track of stop time/frame for later
      fixation.tStop = t;  // not accounting for scr refresh
      fixation.frameNStop = frameN;  // exact frame index
      // update status
      fixation.status = PsychoJS.Status.FINISHED;
      fixation.setAutoDraw(false);
    }
    
    
    // *response* updates
    if (t >= 0.5 && response.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      response.tStart = t;  // (not accounting for frame time here)
      response.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { response.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { response.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { response.clearEvents(); });
    }
    frameRemains = 0.5 + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (response.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      // keep track of stop time/frame for later
      response.tStop = t;  // not accounting for scr refresh
      response.frameNStop = frameN;  // exact frame index
      // update status
      response.status = PsychoJS.Status.FINISHED;
      frameRemains = 0.5 + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
      if (response.status === PsychoJS.Status.STARTED && t >= frameRemains) {
        // keep track of stop time/frame for later
        response.tStop = t;  // not accounting for scr refresh
        response.frameNStop = frameN;  // exact frame index
        // update status
        response.status = PsychoJS.Status.FINISHED;
        response.status = PsychoJS.Status.FINISHED;
          }
        
      }
      
      // if response is active this frame...
      if (response.status === PsychoJS.Status.STARTED) {
        let theseKeys = response.getKeys({
          keyList: typeof ['left','right'] === 'string' ? [['left','right']] : ['left','right'], 
          waitRelease: false
        });
        _response_allKeys = _response_allKeys.concat(theseKeys);
        if (_response_allKeys.length > 0) {
          response.keys = _response_allKeys[_response_allKeys.length - 1].name;  // just the last key pressed
          response.rt = _response_allKeys[_response_allKeys.length - 1].rt;
          response.duration = _response_allKeys[_response_allKeys.length - 1].duration;
          // was this correct?
          if (response.keys == correct_resp) {
              response.corr = 1;
          } else {
              response.corr = 0;
          }
          // a response ends the routine
          continueRoutine = false;
        }
      }
      
      
      // *fish_stim* updates
      if (t >= 0.5 && fish_stim.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        fish_stim.tStart = t;  // (not accounting for frame time here)
        fish_stim.frameNStart = frameN;  // exact frame index
        
        fish_stim.setAutoDraw(true);
      }
      
      
      // if fish_stim is active this frame...
      if (fish_stim.status === PsychoJS.Status.STARTED) {
      }
      
      // check for quit (typically the Esc key)
      if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
        return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
      }
      
      // check if the Routine should terminate
      if (!continueRoutine) {  // a component has requested a forced-end of Routine
        routineForceEnded = true;
        return Scheduler.Event.NEXT;
      }
      
      continueRoutine = false;  // reverts to True if at least one component still running
      for (const thisComponent of trialComponents)
        if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
          continueRoutine = true;
          break;
        }
      
      // refresh the screen if continuing
      if (continueRoutine) {
        return Scheduler.Event.FLIP_REPEAT;
      } else {
        return Scheduler.Event.NEXT;
      }
    };
  }
  
  function trialRoutineEnd(snapshot) {
    return async function () {
      //--- Ending Routine 'trial' ---
      for (const thisComponent of trialComponents) {
        if (typeof thisComponent.setAutoDraw === 'function') {
          thisComponent.setAutoDraw(false);
        }
      }
      psychoJS.experiment.addData('trial.stopped', globalClock.getTime());
      // was no response the correct answer?!
      if (response.keys === undefined) {
        if (['None','none',undefined].includes(correct_resp)) {
           response.corr = 1;  // correct non-response
        } else {
           response.corr = 0;  // failed to respond (incorrectly)
        }
      }
      // store data for current loop
      // update the trial handler
      if (currentLoop instanceof MultiStairHandler) {
        currentLoop.addResponse(response.corr, level);
      }
      psychoJS.experiment.addData('response.keys', response.keys);
      psychoJS.experiment.addData('response.corr', response.corr);
      if (typeof response.keys !== 'undefined') {  // we had a response
          psychoJS.experiment.addData('response.rt', response.rt);
          psychoJS.experiment.addData('response.duration', response.duration);
          routineTimer.reset();
          }
      
      response.stop();
      // the Routine "trial" was not non-slip safe, so reset the non-slip timer
      routineTimer.reset();
      
      // Routines running outside a loop should always advance the datafile row
      if (currentLoop === psychoJS.experiment) {
        psychoJS.experiment.nextEntry(snapshot);
      }
      return Scheduler.Event.NEXT;
    }
  }
  
  function feedbackRoutineBegin(snapshot) {
    return async function () {
      TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
      
      //--- Prepare to start Routine 'feedback' ---
      t = 0;
      frameN = -1;
      continueRoutine = true; // until we're told otherwise
      // keep track of whether this Routine was forcibly ended
      routineForceEnded = false;
      feedbackClock.reset(routineTimer.getTime());
      routineTimer.add(1.000000);
      feedbackMaxDurationReached = false;
      // update component parameters for each repeat
      feedback_text.setText(('Benar!' if response.corr else 'Salah'));
      psychoJS.experiment.addData('feedback.started', globalClock.getTime());
      feedbackMaxDuration = None
      // keep track of which components have finished
      feedbackComponents = [];
      feedbackComponents.push(feedback_text);
      
      for (const thisComponent of feedbackComponents)
        if ('status' in thisComponent)
          thisComponent.status = PsychoJS.Status.NOT_STARTED;
      return Scheduler.Event.NEXT;
    }
  }
  
  function feedbackRoutineEachFrame() {
    return async function () {
      //--- Loop for each frame of Routine 'feedback' ---
      // get current time
      t = feedbackClock.getTime();
      frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
      // update/draw components on each frame
      
      // *feedback_text* updates
      if (t >= 0.0 && feedback_text.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        feedback_text.tStart = t;  // (not accounting for frame time here)
        feedback_text.frameNStart = frameN;  // exact frame index
        
        feedback_text.setAutoDraw(true);
      }
      
      
      // if feedback_text is active this frame...
      if (feedback_text.status === PsychoJS.Status.STARTED) {
      }
      
      frameRemains = 0.0 + 1.0 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
      if (feedback_text.status === PsychoJS.Status.STARTED && t >= frameRemains) {
        // keep track of stop time/frame for later
        feedback_text.tStop = t;  // not accounting for scr refresh
        feedback_text.frameNStop = frameN;  // exact frame index
        // update status
        feedback_text.status = PsychoJS.Status.FINISHED;
        feedback_text.setAutoDraw(false);
      }
      
      // check for quit (typically the Esc key)
      if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
        return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
      }
      
      // check if the Routine should terminate
      if (!continueRoutine) {  // a component has requested a forced-end of Routine
        routineForceEnded = true;
        return Scheduler.Event.NEXT;
      }
      
      continueRoutine = false;  // reverts to True if at least one component still running
      for (const thisComponent of feedbackComponents)
        if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
          continueRoutine = true;
          break;
        }
      
      // refresh the screen if continuing
      if (continueRoutine && routineTimer.getTime() > 0) {
        return Scheduler.Event.FLIP_REPEAT;
      } else {
        return Scheduler.Event.NEXT;
      }
    };
  }
  
  function feedbackRoutineEnd(snapshot) {
    return async function () {
      //--- Ending Routine 'feedback' ---
      for (const thisComponent of feedbackComponents) {
        if (typeof thisComponent.setAutoDraw === 'function') {
          thisComponent.setAutoDraw(false);
        }
      }
      psychoJS.experiment.addData('feedback.stopped', globalClock.getTime());
      if (routineForceEnded) {
          routineTimer.reset();} else if (feedbackMaxDurationReached) {
          feedbackClock.add(feedbackMaxDuration);
      } else {
          feedbackClock.add(1.000000);
      }
      // Routines running outside a loop should always advance the datafile row
      if (currentLoop === psychoJS.experiment) {
        psychoJS.experiment.nextEntry(snapshot);
      }
      return Scheduler.Event.NEXT;
    }
  }
  
  function readyRoutineBegin(snapshot) {
    return async function () {
      TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
      
      //--- Prepare to start Routine 'ready' ---
      t = 0;
      frameN = -1;
      continueRoutine = true; // until we're told otherwise
      // keep track of whether this Routine was forcibly ended
      routineForceEnded = false;
      readyClock.reset();
      routineTimer.reset();
      readyMaxDurationReached = false;
      // update component parameters for each repeat
      ready_key.keys = undefined;
      ready_key.rt = undefined;
      _ready_key_allKeys = [];
      psychoJS.experiment.addData('ready.started', globalClock.getTime());
      readyMaxDuration = None
      // keep track of which components have finished
      readyComponents = [];
      readyComponents.push(ready_text);
      readyComponents.push(ready_key);
      
      for (const thisComponent of readyComponents)
        if ('status' in thisComponent)
          thisComponent.status = PsychoJS.Status.NOT_STARTED;
      return Scheduler.Event.NEXT;
    }
  }
  
  function readyRoutineEachFrame() {
    return async function () {
      //--- Loop for each frame of Routine 'ready' ---
      // get current time
      t = readyClock.getTime();
      frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
      // update/draw components on each frame
      
      // *ready_text* updates
      if (t >= 0.0 && ready_text.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        ready_text.tStart = t;  // (not accounting for frame time here)
        ready_text.frameNStart = frameN;  // exact frame index
        
        ready_text.setAutoDraw(true);
      }
      
      
      // if ready_text is active this frame...
      if (ready_text.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *ready_key* updates
      if (t >= 0.0 && ready_key.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        ready_key.tStart = t;  // (not accounting for frame time here)
        ready_key.frameNStart = frameN;  // exact frame index
        
        // keyboard checking is just starting
        psychoJS.window.callOnFlip(function() { ready_key.clock.reset(); });  // t=0 on next screen flip
        psychoJS.window.callOnFlip(function() { ready_key.start(); }); // start on screen flip
        psychoJS.window.callOnFlip(function() { ready_key.clearEvents(); });
      }
      
      // if ready_key is active this frame...
      if (ready_key.status === PsychoJS.Status.STARTED) {
        let theseKeys = ready_key.getKeys({
          keyList: typeof 'space' === 'string' ? ['space'] : 'space', 
          waitRelease: false
        });
        _ready_key_allKeys = _ready_key_allKeys.concat(theseKeys);
        if (_ready_key_allKeys.length > 0) {
          ready_key.keys = _ready_key_allKeys[_ready_key_allKeys.length - 1].name;  // just the last key pressed
          ready_key.rt = _ready_key_allKeys[_ready_key_allKeys.length - 1].rt;
          ready_key.duration = _ready_key_allKeys[_ready_key_allKeys.length - 1].duration;
          // a response ends the routine
          continueRoutine = false;
        }
      }
      
      // check for quit (typically the Esc key)
      if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
        return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
      }
      
      // check if the Routine should terminate
      if (!continueRoutine) {  // a component has requested a forced-end of Routine
        routineForceEnded = true;
        return Scheduler.Event.NEXT;
      }
      
      continueRoutine = false;  // reverts to True if at least one component still running
      for (const thisComponent of readyComponents)
        if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
          continueRoutine = true;
          break;
        }
      
      // refresh the screen if continuing
      if (continueRoutine) {
        return Scheduler.Event.FLIP_REPEAT;
      } else {
        return Scheduler.Event.NEXT;
      }
    };
  }
  
  function readyRoutineEnd(snapshot) {
    return async function () {
      //--- Ending Routine 'ready' ---
      for (const thisComponent of readyComponents) {
        if (typeof thisComponent.setAutoDraw === 'function') {
          thisComponent.setAutoDraw(false);
        }
      }
      psychoJS.experiment.addData('ready.stopped', globalClock.getTime());
      // update the trial handler
      if (currentLoop instanceof MultiStairHandler) {
        currentLoop.addResponse(ready_key.corr, level);
      }
      psychoJS.experiment.addData('ready_key.keys', ready_key.keys);
      if (typeof ready_key.keys !== 'undefined') {  // we had a response
          psychoJS.experiment.addData('ready_key.rt', ready_key.rt);
          psychoJS.experiment.addData('ready_key.duration', ready_key.duration);
          routineTimer.reset();
          }
      
      ready_key.stop();
      // the Routine "ready" was not non-slip safe, so reset the non-slip timer
      routineTimer.reset();
      
      // Routines running outside a loop should always advance the datafile row
      if (currentLoop === psychoJS.experiment) {
        psychoJS.experiment.nextEntry(snapshot);
      }
      return Scheduler.Event.NEXT;
    }
  }
  
  function break_3RoutineBegin(snapshot) {
    return async function () {
      TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
      
      //--- Prepare to start Routine 'break_3' ---
      t = 0;
      frameN = -1;
      continueRoutine = true; // until we're told otherwise
      // keep track of whether this Routine was forcibly ended
      routineForceEnded = false;
      break_3Clock.reset();
      routineTimer.reset();
      break_3MaxDurationReached = false;
      // update component parameters for each repeat
      break_key.keys = undefined;
      break_key.rt = undefined;
      _break_key_allKeys = [];
      psychoJS.experiment.addData('break_3.started', globalClock.getTime());
      break_3MaxDuration = None
      // keep track of which components have finished
      break_3Components = [];
      break_3Components.push(break_text);
      break_3Components.push(break_key);
      
      for (const thisComponent of break_3Components)
        if ('status' in thisComponent)
          thisComponent.status = PsychoJS.Status.NOT_STARTED;
      return Scheduler.Event.NEXT;
    }
  }
  
  function break_3RoutineEachFrame() {
    return async function () {
      //--- Loop for each frame of Routine 'break_3' ---
      // get current time
      t = break_3Clock.getTime();
      frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
      // update/draw components on each frame
      
      // *break_text* updates
      if (t >= 0.0 && break_text.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        break_text.tStart = t;  // (not accounting for frame time here)
        break_text.frameNStart = frameN;  // exact frame index
        
        break_text.setAutoDraw(true);
      }
      
      
      // if break_text is active this frame...
      if (break_text.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *break_key* updates
      if (t >= 0.0 && break_key.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        break_key.tStart = t;  // (not accounting for frame time here)
        break_key.frameNStart = frameN;  // exact frame index
        
        // keyboard checking is just starting
        psychoJS.window.callOnFlip(function() { break_key.clock.reset(); });  // t=0 on next screen flip
        psychoJS.window.callOnFlip(function() { break_key.start(); }); // start on screen flip
        psychoJS.window.callOnFlip(function() { break_key.clearEvents(); });
      }
      
      // if break_key is active this frame...
      if (break_key.status === PsychoJS.Status.STARTED) {
        let theseKeys = break_key.getKeys({
          keyList: typeof 'space' === 'string' ? ['space'] : 'space', 
          waitRelease: false
        });
        _break_key_allKeys = _break_key_allKeys.concat(theseKeys);
        if (_break_key_allKeys.length > 0) {
          break_key.keys = _break_key_allKeys[_break_key_allKeys.length - 1].name;  // just the last key pressed
          break_key.rt = _break_key_allKeys[_break_key_allKeys.length - 1].rt;
          break_key.duration = _break_key_allKeys[_break_key_allKeys.length - 1].duration;
          // a response ends the routine
          continueRoutine = false;
        }
      }
      
      // check for quit (typically the Esc key)
      if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
        return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
      }
      
      // check if the Routine should terminate
      if (!continueRoutine) {  // a component has requested a forced-end of Routine
        routineForceEnded = true;
        return Scheduler.Event.NEXT;
      }
      
      continueRoutine = false;  // reverts to True if at least one component still running
      for (const thisComponent of break_3Components)
        if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
          continueRoutine = true;
          break;
        }
      
      // refresh the screen if continuing
      if (continueRoutine) {
        return Scheduler.Event.FLIP_REPEAT;
      } else {
        return Scheduler.Event.NEXT;
      }
    };
  }
  
  function break_3RoutineEnd(snapshot) {
    return async function () {
      //--- Ending Routine 'break_3' ---
      for (const thisComponent of break_3Components) {
        if (typeof thisComponent.setAutoDraw === 'function') {
          thisComponent.setAutoDraw(false);
        }
      }
      psychoJS.experiment.addData('break_3.stopped', globalClock.getTime());
      // update the trial handler
      if (currentLoop instanceof MultiStairHandler) {
        currentLoop.addResponse(break_key.corr, level);
      }
      psychoJS.experiment.addData('break_key.keys', break_key.keys);
      if (typeof break_key.keys !== 'undefined') {  // we had a response
          psychoJS.experiment.addData('break_key.rt', break_key.rt);
          psychoJS.experiment.addData('break_key.duration', break_key.duration);
          routineTimer.reset();
          }
      
      break_key.stop();
      // the Routine "break_3" was not non-slip safe, so reset the non-slip timer
      routineTimer.reset();
      
      // Routines running outside a loop should always advance the datafile row
      if (currentLoop === psychoJS.experiment) {
        psychoJS.experiment.nextEntry(snapshot);
      }
      return Scheduler.Event.NEXT;
    }
  }
  
  function endRoutineBegin(snapshot) {
    return async function () {
      TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
      
      //--- Prepare to start Routine 'end' ---
      t = 0;
      frameN = -1;
      continueRoutine = true; // until we're told otherwise
      // keep track of whether this Routine was forcibly ended
      routineForceEnded = false;
      endClock.reset(routineTimer.getTime());
      routineTimer.add(3.000000);
      endMaxDurationReached = false;
      // update component parameters for each repeat
      psychoJS.experiment.addData('end.started', globalClock.getTime());
      endMaxDuration = None
      // keep track of which components have finished
      endComponents = [];
      endComponents.push(end_text);
      
      for (const thisComponent of endComponents)
        if ('status' in thisComponent)
          thisComponent.status = PsychoJS.Status.NOT_STARTED;
      return Scheduler.Event.NEXT;
    }
  }
  
  function endRoutineEachFrame() {
    return async function () {
      //--- Loop for each frame of Routine 'end' ---
      // get current time
      t = endClock.getTime();
      frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
      // update/draw components on each frame
      
      // *end_text* updates
      if (t >= 0.0 && end_text.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        end_text.tStart = t;  // (not accounting for frame time here)
        end_text.frameNStart = frameN;  // exact frame index
        
        end_text.setAutoDraw(true);
      }
      
      
      // if end_text is active this frame...
      if (end_text.status === PsychoJS.Status.STARTED) {
      }
      
      frameRemains = 0.0 + 3.0 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
      if (end_text.status === PsychoJS.Status.STARTED && t >= frameRemains) {
        // keep track of stop time/frame for later
        end_text.tStop = t;  // not accounting for scr refresh
        end_text.frameNStop = frameN;  // exact frame index
        // update status
        end_text.status = PsychoJS.Status.FINISHED;
        end_text.setAutoDraw(false);
      }
      
      // check for quit (typically the Esc key)
      if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
        return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
      }
      
      // check if the Routine should terminate
      if (!continueRoutine) {  // a component has requested a forced-end of Routine
        routineForceEnded = true;
        return Scheduler.Event.NEXT;
      }
      
      continueRoutine = false;  // reverts to True if at least one component still running
      for (const thisComponent of endComponents)
        if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
          continueRoutine = true;
          break;
        }
      
      // refresh the screen if continuing
      if (continueRoutine && routineTimer.getTime() > 0) {
        return Scheduler.Event.FLIP_REPEAT;
      } else {
        return Scheduler.Event.NEXT;
      }
    };
  }
  
  function endRoutineEnd(snapshot) {
    return async function () {
      //--- Ending Routine 'end' ---
      for (const thisComponent of endComponents) {
        if (typeof thisComponent.setAutoDraw === 'function') {
          thisComponent.setAutoDraw(false);
        }
      }
      psychoJS.experiment.addData('end.stopped', globalClock.getTime());
      if (routineForceEnded) {
          routineTimer.reset();} else if (endMaxDurationReached) {
          endClock.add(endMaxDuration);
      } else {
          endClock.add(3.000000);
      }
      // Routines running outside a loop should always advance the datafile row
      if (currentLoop === psychoJS.experiment) {
        psychoJS.experiment.nextEntry(snapshot);
      }
      return Scheduler.Event.NEXT;
    }
  }
  
  function importConditions(currentLoop) {
    return async function () {
      psychoJS.importAttributes(currentLoop.getCurrentTrial());
      return Scheduler.Event.NEXT;
      };
  }
  
  async function quitPsychoJS(message, isCompleted) {
    // Check for and save orphaned data
    if (psychoJS.experiment.isEntryEmpty()) {
      psychoJS.experiment.nextEntry();
    }
    psychoJS.window.close();
    psychoJS.quit({message: message, isCompleted: isCompleted});
    
    return Scheduler.Event.QUIT;
  }
