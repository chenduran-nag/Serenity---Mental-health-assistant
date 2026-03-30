import { useEffect, useRef } from 'react';
import PropTypes from 'prop-types';

import styles from './ParticleCanvas.module.css';

const COLORS = ['#00e5cc', '#6e44ff', '#ff2d9b', '#8cf9ff'];
const PARTICLE_COUNT = 80;

function createParticles(width, height) {
  return Array.from({ length: PARTICLE_COUNT }, () => ({
    x: Math.random() * width,
    y: Math.random() * height,
    radius: 2 + Math.random() * 6,
    alpha: 0.3 + Math.random() * 0.4,
    speedX: (Math.random() - 0.5) * 0.25,
    speedY: (Math.random() - 0.5) * 0.25,
    color: COLORS[Math.floor(Math.random() * COLORS.length)],
  }));
}

export default function ParticleCanvas({ pulseSeed }) {
  const canvasRef = useRef(null);
  const pulseUntilRef = useRef(0);

  useEffect(() => {
    pulseUntilRef.current = performance.now() + 1500;
  }, [pulseSeed]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return undefined;
    }

    const context = canvas.getContext('2d');
    let animationFrame = 0;
    let particles = [];

    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      particles = createParticles(canvas.width, canvas.height);
    }

    function draw(now) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const isPulsing = now < pulseUntilRef.current;
      const pulseStrength = isPulsing ? (pulseUntilRef.current - now) / 1500 : 0;

      for (const particle of particles) {
        particle.x += particle.speedX;
        particle.y += particle.speedY;

        if (particle.x < -20) particle.x = canvas.width + 20;
        if (particle.x > canvas.width + 20) particle.x = -20;
        if (particle.y < -20) particle.y = canvas.height + 20;
        if (particle.y > canvas.height + 20) particle.y = -20;

        const dx = particle.x - centerX;
        const dy = particle.y - centerY;
        const distance = Math.max(Math.hypot(dx, dy), 1);
        const bloomOffset = isPulsing ? (1 - pulseStrength) * 4 : 0;
        const renderX = particle.x + (dx / distance) * bloomOffset;
        const renderY = particle.y + (dy / distance) * bloomOffset;

        context.beginPath();
        context.fillStyle = particle.color;
        context.globalAlpha = particle.alpha;
        context.shadowColor = particle.color;
        context.shadowBlur = 18 + particle.radius * (isPulsing ? 4 : 2);
        context.arc(renderX, renderY, particle.radius + (isPulsing ? 1.5 : 0), 0, Math.PI * 2);
        context.fill();
      }

      context.globalAlpha = 1;
      animationFrame = window.requestAnimationFrame(draw);
    }

    resize();
    window.addEventListener('resize', resize);
    animationFrame = window.requestAnimationFrame(draw);

    return () => {
      window.cancelAnimationFrame(animationFrame);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return <canvas className={styles.canvas} ref={canvasRef} />;
}

ParticleCanvas.propTypes = {
  pulseSeed: PropTypes.number.isRequired,
};
