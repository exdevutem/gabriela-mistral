import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const SUAVIZADO = 14;   // cuanto más alto, más rápido llega la boca a su forma
const DERIVA = 0.035;   // amplitud del vaivén de cabeza, en radianes

export async function crearAvatar(canvas) {
  const escena = new THREE.Scene();
  escena.background = new THREE.Color(0x14100e);

  const camara = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camara.position.set(0, 0.05, 4.6);

  const render = new THREE.WebGLRenderer({ canvas, antialias: true });
  render.setPixelRatio(Math.min(devicePixelRatio, 2));

  // Tres luces: una principal cálida, un relleno frío y un contraluz que
  // recorta la silueta. Sin el contraluz la cabeza se funde con el fondo.
  const principal = new THREE.DirectionalLight(0xfff0dd, 2.6);
  principal.position.set(2, 3, 4);
  const relleno = new THREE.DirectionalLight(0x93b4d8, 0.8);
  relleno.position.set(-3, -0.5, 2);
  const contra = new THREE.DirectionalLight(0xffd9a8, 1.4);
  contra.position.set(-1.5, 1.5, -3);
  escena.add(principal, relleno, contra, new THREE.AmbientLight(0x40382f, 1.2));

  const gltf = await new GLTFLoader().loadAsync('/assets/gabriela.glb');
  const raiz = gltf.scene;
  escena.add(raiz);

  let malla = null;
  raiz.traverse((o) => { if (o.isMesh && o.morphTargetInfluences) malla = o; });
  if (!malla) throw new Error('el GLB no trae morph targets');
  const indices = malla.morphTargetDictionary;
  const nombres = Object.keys(indices);

  const controles = new OrbitControls(camara, canvas);
  controles.target.set(0, 0, 0);
  controles.enablePan = false;
  controles.minDistance = 2.5;
  controles.maxDistance = 8;

  const objetivo = new Float32Array(malla.morphTargetInfluences.length);
  let pista = null, audio = null, cursor = 0, manual = false;

  function ajustar() {
    const { clientWidth: w, clientHeight: h } = canvas;
    if (canvas.width !== w || canvas.height !== h) {
      render.setSize(w, h, false);
      camara.aspect = w / h;
      camara.updateProjectionMatrix();
    }
  }

  const reloj = new THREE.Clock();
  function bucle() {
    requestAnimationFrame(bucle);
    ajustar();
    const dt = reloj.getDelta(), t = reloj.getElapsedTime();

    if (pista && audio && !audio.paused) {
      const ahora = audio.currentTime;
      while (cursor + 1 < pista.length && pista[cursor + 1][0] <= ahora) cursor++;
      const [, visema, peso] = pista[cursor];
      objetivo.fill(0);
      if (indices[visema] !== undefined) objetivo[indices[visema]] = peso;
    } else if (pista && audio && audio.ended) {
      objetivo.fill(0);
      pista = null;
    }

    if (!manual) {
      const k = 1 - Math.exp(-SUAVIZADO * dt);  // lerp estable ante saltos de fps
      for (let i = 0; i < objetivo.length; i++) {
        malla.morphTargetInfluences[i] += (objetivo[i] - malla.morphTargetInfluences[i]) * k;
      }
    }

    // Micro-vida: sin esto la cabeza se lee como un objeto, no como alguien
    // escuchando. Frecuencias primas para que el vaivén no se sienta cíclico.
    raiz.rotation.y = Math.sin(t * 0.31) * DERIVA + Math.sin(t * 0.13) * DERIVA * 0.6;
    raiz.rotation.x = Math.sin(t * 0.23) * DERIVA * 0.5;
    raiz.position.y = Math.sin(t * 0.7) * 0.006;  // respiración

    render.render(escena, camara);
  }
  bucle();

  return {
    nombres,
    malla,  // expuesta para inspección desde la página de ajuste
    hablar(audioBase64, visemas) {
      manual = false;
      pista = visemas;
      cursor = 0;
      audio = new Audio('data:audio/wav;base64,' + audioBase64);
      return audio.play();
    },
    /** Para la página de ajuste: fija un morph target a mano. */
    fijar(nombre, peso) {
      manual = true;
      pista = null;
      malla.morphTargetInfluences[indices[nombre]] = peso;
    },
  };
}
