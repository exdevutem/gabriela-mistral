import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const SUAVIZADO = 14;   // cuanto más alto, más rápido llega la boca a su forma
const DERIVA = 0.035;   // amplitud del vaivén de cabeza, en radianes

export async function crearAvatar(canvas) {
  const escena = new THREE.Scene();
  escena.background = new THREE.Color(0x14100e);

  const camara = new THREE.PerspectiveCamera(32, 1, 0.01, 100);

  const render = new THREE.WebGLRenderer({ canvas, antialias: true });
  render.setPixelRatio(Math.min(devicePixelRatio, 2));

  // Iluminación deliberadamente plana. La textura es una fotografía y ya trae
  // sus sombras horneadas: sumarle un esquema de tres puntos las duplica y
  // ennegrece medio rostro. Domina la ambiental, y las direccionales sólo
  // insinúan el volumen y recortan la silueta contra el fondo.
  const principal = new THREE.DirectionalLight(0xfff4e6, 0.85);
  principal.position.set(2, 3, 4);
  const relleno = new THREE.DirectionalLight(0xbcd0e8, 0.35);
  relleno.position.set(-3, -0.5, 2);
  const contra = new THREE.DirectionalLight(0xffd9a8, 0.55);
  contra.position.set(-1.5, 1.5, -3);
  escena.add(principal, relleno, contra, new THREE.AmbientLight(0xfff2e2, 2.1));

  const gltf = await new GLTFLoader().loadAsync('/assets/gabriela.glb');
  const raiz = gltf.scene;
  escena.add(raiz);

  let malla = null;
  raiz.traverse((o) => {
    if (!o.isMesh) return;
    if (o.morphTargetInfluences) malla = o;
    // GLTFLoader no activa vertexColors si el material se creó sin ellos
    if (o.geometry.attributes.color) o.material.vertexColors = true;
  });
  if (!malla) throw new Error('el GLB no trae morph targets');
  const indices = malla.morphTargetDictionary;
  const nombres = Object.keys(indices);

  // Encuadre automático: FLAME viene en metros (la cabeza mide unos 20 cm) y el
  // esferoide de prueba medía ~2 unidades. Deducir la distancia del propio modelo
  // evita una constante que hay que recordar cambiar con cada malla nueva.
  const caja = new THREE.Box3().setFromObject(raiz);
  const medida = caja.getSize(new THREE.Vector3());
  const centro = caja.getCenter(new THREE.Vector3());
  const alcance = Math.max(medida.x, medida.y, medida.z);
  const distancia = alcance / (2 * Math.tan((camara.fov * Math.PI / 180) / 2)) * 1.6;
  raiz.position.sub(centro);                      // centrar en el origen
  const alturaBase = raiz.position.y;             // la respiración se suma a esto
  camara.position.set(0, alcance * 0.04, distancia);
  camara.updateProjectionMatrix();

  const controles = new OrbitControls(camara, canvas);
  controles.target.set(0, 0, 0);
  controles.enablePan = false;
  controles.minDistance = distancia * 0.4;
  controles.maxDistance = distancia * 2.5;

  const objetivo = new Float32Array(malla.morphTargetInfluences.length);
  let pista = null, cursor = 0, manual = false;
  const cola = [];      // frases pendientes; el servidor las manda de a una
  let encadenando = false;

  // Un solo elemento de audio para toda la sesión, en vez de uno por frase.
  // Safari solo deja sonar el audio que se arrancó dentro de un gesto del
  // usuario, y la voz tarda medio minuto en llegar: para entonces el permiso
  // del clic ya caducó. El permiso queda atado al *elemento*, así que
  // desbloqueando este al enviar la pregunta, todas las frases lo heredan.
  const audio = new Audio();
  audio.addEventListener('ended', () => { siguiente(); });
  // 1 ms de silencio: lo mínimo que se puede reproducir para ganar el permiso.
  const SILENCIO = 'data:audio/wav;base64,UklGRjQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YRAAAAAAAAAAAAAAAAAAAAAAAAAA';
  let desbloqueado = false;

  function ajustar() {
    const { clientWidth: w, clientHeight: h } = canvas;
    if (canvas.width !== w || canvas.height !== h) {
      render.setSize(w, h, false);
      camara.aspect = w / h;
      camara.updateProjectionMatrix();
    }
  }

  function siguiente() {
    const parte = cola.shift();
    if (!parte) { encadenando = false; pista = null; objetivo.fill(0); return Promise.resolve(); }
    const [audioBase64, visemas] = parte;
    pista = visemas;
    cursor = 0;
    audio.src = 'data:audio/wav;base64,' + audioBase64;
    return audio.play();
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
      if (!cola.length) pista = null;  // con más frases en cola, 'ended' encadena
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
    raiz.position.y = alturaBase + Math.sin(t * 0.7) * alcance * 0.003;  // respiración

    render.render(escena, camara);
  }
  bucle();

  return {
    nombres,
    malla,  // expuesta para inspección desde la página de ajuste
    /**
     * Encola una frase. El servidor las manda una a una según las sintetiza, así
     * que ella empieza a hablar con la primera mientras llegan las demás; si se
     * reprodujera cada una al recibirla, se pisarían.
     */
    hablar(audioBase64, visemas) {
      manual = false;
      cola.push([audioBase64, visemas]);
      if (encadenando) return Promise.resolve();
      encadenando = true;
      return siguiente();
    },
    /**
     * Gana el permiso del navegador para reproducir audio, reproduciendo un
     * silencio. Hay que llamarlo **desde un gesto del usuario** —el envío de la
     * pregunta—, porque para cuando llega la voz ese permiso ya caducó.
     */
    desbloquear() {
      if (desbloqueado || !audio.paused) return;  // si ya suena, el permiso está
      audio.src = SILENCIO;
      // AbortError es lo normal, no un fallo: la primera frase llega y cambia
      // el src antes de que el silencio termine. El permiso ya se ganó, y darlo
      // por perdido haría que el siguiente envío pisara el audio en curso.
      audio.play().then(
        () => { desbloqueado = true; },
        (e) => { desbloqueado = e.name === 'AbortError'; },
      );
    },

    /** Para la página de ajuste: fija un morph target a mano. */
    fijar(nombre, peso) {
      manual = true;
      pista = null;
      malla.morphTargetInfluences[indices[nombre]] = peso;
    },
  };
}
