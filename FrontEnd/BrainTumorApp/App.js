import React, { useState } from "react";
import { StyleSheet, Text, View, Button, Image, ActivityIndicator } from "react-native";
import * as ImagePicker from "expo-image-picker";
import axios from "axios";

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  // Pick image from gallery
  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
      uploadImage(result.assets[0].uri);
    }
  };

  // Upload to FastAPI backend
  const uploadImage = async (uri) => {
    setLoading(true);
    setPrediction(null);

    let formData = new FormData();
    formData.append("file", {
      uri,
      type: "image/jpeg",
      name: "upload.jpg",
    });

    try {
      const response = await axios.post("http://10.0.2.2:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPrediction(response.data);
    } catch (error) {
      console.error(error);
      alert("Error connecting to backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Button title="Pick MRI Image" onPress={pickImage} />

      {selectedImage && <Image source={{ uri: selectedImage }} style={styles.image} />}

      {loading && <ActivityIndicator size="large" color="#0000ff" />}

      {prediction && (
        <Text style={styles.result}>
          Prediction: {prediction.prediction} {"\n"}
          Confidence: {(prediction.confidence * 100).toFixed(2)}%
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  image: {
    width: 250,
    height: 250,
    marginVertical: 20,
  },
  result: {
    marginTop: 20,
    fontSize: 18,
    fontWeight: "bold",
    textAlign: "center",
  },
});
